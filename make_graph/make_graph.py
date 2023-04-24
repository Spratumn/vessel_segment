import numpy as np
import os
import sys
import networkx as nx
import pickle as pkl
import multiprocessing
import matplotlib.pyplot as plt
import argparse
import skfmm
import cv2

from .bwmorph import bwmorph
from .visualize import visualize_graph

sys.path.append('../config')

from config import DATASET_CONFIGS


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Make a graph db')
    parser.add_argument('--dataset', default='Artery',
                        help='Dataset dir to use', type=str)
    parser.add_argument('--multiprocess', default=0,
                        help='use the python multiprocessing module', type=int)
    parser.add_argument('--source_type', default='gt',
                        help='Source to be used: Can be result or gt', type=str)
    parser.add_argument('--debug',
                        action="store_true",
                        help='show the travel time')
    parser.add_argument('--win_size', default=8,
                        help='Window size for srns', type=int) # for srns # [4,8,16]
    parser.add_argument('--edge_method', default='geo_dist',
                        help='Edge construction method: Can be geo_dist or eu_dist', type=str)
    parser.add_argument('--edge_dist_thresh', default=10,
                        help='Distance threshold for edge construction', type=float) # [10,20,40]
    args = parser.parse_args()
    return args


def generate_graph_using_srns(args_tuple):
    img_name, graph_root_dir, params = args_tuple
    win_size_str = '%.2d_%.2d'%(params.win_size, params.edge_dist_thresh)

    if params.source_type == 'gt': win_size_str = win_size_str + '_gt'

    cur_filename = os.path.basename(img_name)
    print('Processing ' + cur_filename + '...')
    cur_image_path = img_name + DATASET_CONFIGS[params.dataset].image_suffix
    cur_gt_path = img_name + DATASET_CONFIGS[params.dataset].label_suffix

    if params.source_type=='gt':
        cur_res_prob_path = cur_gt_path
    else:
        cur_res_prob_path = os.path.join(graph_root_dir, cur_filename + '_prob.png')
    assert os.path.exists(cur_res_prob_path), f'{cur_res_prob_path} not exists!'

    cur_vis_res_im_savepath = os.path.join(graph_root_dir,
                                           cur_filename + '_' + win_size_str + '_vis_graph_res_on_im.png')
    cur_vis_res_mask_savepath = os.path.join(graph_root_dir,
                                             cur_filename + '_' + win_size_str + '_vis_graph_res_on_mask.png')
    cur_res_graph_savepath = os.path.join(graph_root_dir,
                                          cur_filename+'_' + win_size_str + '.graph_res')

    H = DATASET_CONFIGS[params.dataset].len_y
    W = DATASET_CONFIGS[params.dataset].len_x
    cur_image = np.zeros((H, W, 3), dtype='uint8')
    cur_gt = np.zeros((H, W), dtype='uint8')
    cur_vessel = np.zeros((H, W), dtype='uint8')

    if os.path.exists(cur_image_path):
        image = cv2.cvtColor(cv2.imread(cur_image_path), cv2.COLOR_BGR2RGB)
        if image.shape[0] != H or image.shape[1] != W:
            image = cv2.resize(image, (W, H))
        cur_image[:, :] = image
    if os.path.exists(cur_gt_path):
        gt_mask = cv2.imread(cur_gt_path, 0)
        if gt_mask.shape[0] != H or gt_mask.shape[1] != W:
            gt_mask = cv2.resize(gt_mask, (W, H))
        cur_gt[:, :] = gt_mask

    cur_gt = cur_gt.astype(float) / 255
    cur_gt = cur_gt >= 0.5

    vesselness = cv2.imread(cur_res_prob_path, 0)
    if vesselness.shape[0] != H or vesselness.shape[1] != W:
        vesselness = cv2.resize(vesselness, (W, H))
    cur_vessel[:, :] = vesselness
    cur_vessel = cur_vessel.astype(float) / 255

    # find local maxima
    y_quan = range(0, H, params.win_size)
    y_quan = sorted(list(set(y_quan) | set([H])))
    x_quan = range(0, W, params.win_size)
    x_quan = sorted(list(set(x_quan) | set([W])))

    max_val = []
    max_pos = []
    for y_idx in range(len(y_quan)-1):
        for x_idx in range(len(x_quan)-1):
            cur_patch = cur_vessel[y_quan[y_idx]:y_quan[y_idx+1], x_quan[x_idx]:x_quan[x_idx+1]]
            if np.sum(cur_patch)==0:
                max_val.append(0)
                max_pos.append((y_quan[y_idx]+cur_patch.shape[0]/2, x_quan[x_idx]+cur_patch.shape[1]/2))
            else:
                max_val.append(np.amax(cur_patch))
                temp = np.unravel_index(cur_patch.argmax(), cur_patch.shape)
                max_pos.append((y_quan[y_idx]+temp[0], x_quan[x_idx]+temp[1]))

    graph = nx.Graph()
    # add nodes
    for node_idx, (node_y, node_x) in enumerate(max_pos):
        graph.add_node(node_idx, kind='MP', y=node_y, x=node_x, label=node_idx)
        # print('node label', node_idx, 'pos', (node_y, node_x), 'added')

    speed = cur_vessel
    if params.source_type == 'gt':
        speed = bwmorph(speed, 'dilate', n_iter=1)
        speed = speed.astype(float)

    edge_dist_thresh_sq = params.edge_dist_thresh**2
    node_list = list(graph.nodes)

    for i, n in enumerate(node_list):
        x_i = int(graph.node[n]['x'])
        y_i = int(graph.node[n]['y'])
        if speed[y_i, x_i] == 0: continue
        neighbor = speed[max(0,y_i-1):min(H, y_i+2), max(0,x_i-1):min(W, x_i+2)]
        if np.mean(neighbor) < 0.1: continue
        if params.edge_method == 'geo_dist':
            phi = np.ones_like(speed)
            phi[y_i, x_i] = -1
            tt = skfmm.travel_time(phi, speed, narrow=params.edge_dist_thresh)
            if params.debug:
                plt.figure()
                plt.imshow(tt, interpolation='nearest')
                plt.show()
                plt.cla()
                plt.clf()
                plt.close()
            for n_comp in node_list[i+1:]:
                n_x_i = int(graph.node[n_comp]['x'])
                n_y_i = int(graph.node[n_comp]['y'])
                geo_dist = tt[n_y_i, n_x_i]
                if geo_dist < params.edge_dist_thresh:
                    graph.add_edge(n, n_comp, weight=params.edge_dist_thresh/(params.edge_dist_thresh+geo_dist))
                    # print('An edge BTWN', 'node', n, '&', n_comp, 'is constructed')
        elif params.edge_method=='eu_dist':
            for n_comp in node_list[i+1:]:
                n_x_i = int(graph.node[n_comp]['x'])
                n_y_i = int(graph.node[n_comp]['y'])
                eu_dist = (n_y_i-y_i)**2 + (n_x_i-x_i)**2
                if eu_dist < edge_dist_thresh_sq:
                    graph.add_edge(n, n_comp, weight=1.)
                    # print('An edge BTWN', 'node', n, '&', n_comp, 'is constructed')
        else:
            raise NotImplementedError

    # visualize the constructed graph
    if os.path.exists(cur_image_path):
        visualize_graph(image, graph, show_graph=False,
                        save_graph=True,
                        num_nodes_each_type=[0, graph.number_of_nodes()],
                        save_path=cur_vis_res_im_savepath)
    if os.path.exists(cur_gt_path):
        visualize_graph(gt_mask, graph, show_graph=False,
                        save_graph=True,
                        num_nodes_each_type=[0, graph.number_of_nodes()],
                        save_path=cur_vis_res_mask_savepath)
    # save as files
    nx.write_gpickle(graph, cur_res_graph_savepath, protocol=pkl.HIGHEST_PROTOCOL)
    graph.clear()
    print(cur_filename, ' finished!')


def make_graph(args):
    assert args.dataset in DATASET_CONFIGS
    assert os.path.exists(os.path.join('datasets', args.dataset))
    train_set_txt_path = os.path.join('datasets', args.dataset, 'train.txt')
    test_set_txt_path = os.path.join('datasets', args.dataset, 'test.txt')
    graph_root_dir = os.path.join('datasets', args.dataset, 'graph')

    if args.source_type != 'gt':
        assert os.path.exists(graph_root_dir)
    else:
        if not os.path.exists(graph_root_dir): os.mkdir(graph_root_dir)

    with open(train_set_txt_path) as f:
        train_image_paths = [x.strip() for x in f.readlines()]
    with open(test_set_txt_path) as f:
        test_image_paths = [x.strip() for x in f.readlines()]

    len_train = len(train_image_paths)
    len_test = len(test_image_paths)

    func_arg_train = map(lambda x: (train_image_paths[x], graph_root_dir, args), range(len_train))
    func_arg_test = map(lambda x: (test_image_paths[x], graph_root_dir, args), range(len_test))

    if args.multiprocess > 1:
        pool = multiprocessing.Pool(processes=args.multiprocess)
        pool.map(generate_graph_using_srns, func_arg_train)
        pool.map(generate_graph_using_srns, func_arg_test)
        pool.terminate()
    else:
        for func_arg in func_arg_train:
            generate_graph_using_srns(func_arg)
        for func_arg in func_arg_test:
            generate_graph_using_srns(func_arg)


if __name__ == '__main__':
    make_graph(parse_args())
