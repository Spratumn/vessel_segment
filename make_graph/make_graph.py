import numpy as np
import os
import networkx as nx
import pickle as pkl
import multiprocessing
import matplotlib.pyplot as plt
import argparse
import skfmm
import cv2

from .bwmorph import bwmorph
from .visualize import visualize_graph


im_ext = '.bmp'
label_ext = '.bmp'
len_y = 400
len_x = 400


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Make a graph db')
    parser.add_argument('--dataset_dir', default='dataset/GT_Artery',
                        help='Dataset dir to use', type=str)
    parser.add_argument('--use_multiprocessing', default=False,
                        help='Whether to use the python multiprocessing module', type=bool)
    parser.add_argument('--source_type', default='gt',
                        help='Source to be used: Can be result or gt', type=str)
    parser.add_argument('--debug', default=False,
                        help='show the travel time', type=bool)
    parser.add_argument('--win_size', default=4,
                        help='Window size for srns', type=int) # for srns # [4,8,16]
    parser.add_argument('--edge_method', default='geo_dist',
                        help='Edge construction method: Can be geo_dist or eu_dist', type=str)
    parser.add_argument('--edge_dist_thresh', default=10,
                        help='Distance threshold for edge construction', type=float) # [10,20,40]
    args = parser.parse_args()
    return args


def generate_graph_using_srns(args_tuple):
    img_name, cnn_result_root_dir, params = args_tuple
    win_size_str = '%.2d_%.2d'%(params.win_size,params.edge_dist_thresh)

    if params.source_type == 'gt':
        win_size_str = win_size_str + '_gt'

    cur_filename = os.path.basename(img_name)
    print('processing ' + cur_filename)
    cur_im_path = img_name + im_ext
    cur_gt_mask_path = img_name + label_ext

    if params.source_type=='gt':
        cur_res_prob_path = cur_gt_mask_path
    else:
        cur_res_prob_path = os.path.join(cnn_result_root_dir, cur_filename + '_prob.png')

    cur_vis_res_im_savepath = os.path.join(cnn_result_root_dir, cur_filename + '_' + win_size_str + '_vis_graph_res_on_im.png')
    cur_vis_res_mask_savepath = os.path.join(cnn_result_root_dir, cur_filename + '_' + win_size_str + '_vis_graph_res_on_mask.png')
    cur_res_graph_savepath = os.path.join(cnn_result_root_dir, cur_filename+'_' + win_size_str + '.graph_res')

    im = cv2.imread(cur_im_path)
    gt_mask = cv2.imread(cur_gt_mask_path, 0)
    gt_mask = gt_mask.astype(float) / 255
    gt_mask = gt_mask >= 0.5

    vesselness = cv2.imread(cur_res_prob_path, 0)
    vesselness = vesselness.astype(float) / 255

    # find local maxima
    im_y = im.shape[0]
    im_x = im.shape[1]
    y_quan = range(0, im_y, params.win_size)
    y_quan = sorted(list(set(y_quan) | set([im_y])))
    x_quan = range(0, im_x, params.win_size)
    x_quan = sorted(list(set(x_quan) | set([im_x])))

    max_val = []
    max_pos = []
    for y_idx in range(len(y_quan)-1):
        for x_idx in range(len(x_quan)-1):
            cur_patch = vesselness[y_quan[y_idx]:y_quan[y_idx+1], x_quan[x_idx]:x_quan[x_idx+1]]
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
        print('node label', node_idx, 'pos', (node_y, node_x), 'added')

    speed = vesselness
    if params.source_type=='gt':
        speed = bwmorph(speed, 'dilate', n_iter=1)
        speed = speed.astype(float)

    edge_dist_thresh_sq = params.edge_dist_thresh**2
    node_list = list(graph.nodes)

    for i, n in enumerate(node_list):
        x_i = int(graph.node[n]['x'])
        y_i = int(graph.node[n]['y'])
        if speed[y_i, x_i] == 0: continue
        neighbor = speed[max(0,y_i-1):min(im_y,y_i+2), max(0,x_i-1):min(im_x,x_i+2)]
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
                    print('An edge BTWN', 'node', n, '&', n_comp, 'is constructed')
        elif params.edge_method=='eu_dist':
            for n_comp in node_list[i+1:]:
                n_x_i = int(graph.node[n_comp]['x'])
                n_y_i = int(graph.node[n_comp]['y'])
                eu_dist = (n_y_i-y_i)**2 + (n_x_i-x_i)**2
                if eu_dist < edge_dist_thresh_sq:
                    graph.add_edge(n, n_comp, weight=1.)
                    print('An edge BTWN', 'node', n, '&', n_comp, 'is constructed')
        else:
            raise NotImplementedError

    # visualize the constructed graph
    visualize_graph(im, graph, show_graph=True,
                    save_graph=True,
                    num_nodes_each_type=[0, graph.number_of_nodes()],
                    save_path=cur_vis_res_im_savepath)
    visualize_graph(gt_mask, graph, show_graph=False,
                    save_graph=True,
                    num_nodes_each_type=[0, graph.number_of_nodes()],
                    save_path=cur_vis_res_mask_savepath)
    # save as files
    nx.write_gpickle(graph, cur_res_graph_savepath, protocol=pkl.HIGHEST_PROTOCOL)
    graph.clear()


def make_graph(args):
    assert os.path.exists(args.dataset_dir)
    train_set_txt_path = os.path.join(args.dataset_dir, 'train.txt')
    test_set_txt_path = os.path.join(args.dataset_dir, 'test.txt')
    cnn_result_root_dir = os.path.join(args.dataset_dir, 'graph_cnn')

    if args.source_type != 'gt':
        assert os.path.exists(cnn_result_root_dir)
    else:
        if not os.path.exists(cnn_result_root_dir): os.mkdir(cnn_result_root_dir)

    with open(train_set_txt_path) as f:
        train_image_paths = [x.strip() for x in f.readlines()]
    with open(test_set_txt_path) as f:
        test_image_paths = [x.strip() for x in f.readlines()]

    len_train = len(train_image_paths)
    len_test = len(test_image_paths)

    func_arg_train = map(lambda x: (train_image_paths[x], cnn_result_root_dir, args), range(len_train))
    func_arg_test = map(lambda x: (test_image_paths[x], cnn_result_root_dir, args), range(len_test))

    if args.use_multiprocessing:
        pool = multiprocessing.Pool(processes=16)
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
