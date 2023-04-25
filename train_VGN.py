import numpy as np
import os
import argparse
import skimage.io
import networkx as nx
import pickle as pkl
import multiprocessing
import sys
import skfmm


import config as cfg
from models.vgn import VesselSegmVGN
import util



def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a vessel_segm_vgn network')
    parser.add_argument('--dataset', default='Artery', help='Dataset to use: Can be DRIVE or STARE or CHASE_DB1 or HRF', type=str)
    parser.add_argument('--use_multiprocessing', default=True, help='Whether to use the python multiprocessing module', type=bool)
    parser.add_argument('--multiprocessing_num_proc', default=36, help='Number of CPU processes to use', type=int)
    parser.add_argument('--win_size', default=8, help='Window size for srns', type=int) # for srns # [4,8,16]
    parser.add_argument('--edge_type', default='srns_geo_dist_binary', help='Graph edge type: Can be srns_geo_dist_binary or srns_geo_dist_weighted', type=str)
    parser.add_argument('--edge_geo_dist_thresh', default=10, help='Threshold for geodesic distance', type=float)
    parser.add_argument('--pretrained_model', default=None, help='Path for a pretrained model(.ckpt)', type=str)
    parser.add_argument('--save_root', default='log', help='root path to save trained models and test results', type=str)

    ### cnn module related ###
    parser.add_argument('--cnn_model', default='driu', help='CNN model to use', type=str)
    parser.add_argument('--cnn_loss_on', default=True, help='Whether to use a cnn loss for training', type=bool)

    ### gnn module related ###
    parser.add_argument('--gnn_loss_on', default=True, help='Whether to use a gnn loss for training', type=bool)
    parser.add_argument('--gnn_loss_weight', default=1., help='Relative weight on the gnn loss', type=float)
    parser.add_argument('--gnn_feat_dropout_prob', default=0.5, help='Dropout prob. for feat. in gnn layers', type=float)
    parser.add_argument('--gnn_att_dropout_prob', default=0.5, help='Dropout prob. for att. in gnn layers', type=float)
    # gat #
    parser.add_argument('--gat_n_heads', default=[4,4], help='Numbers of heads in each layer', type=list)
    parser.add_argument('--gat_hid_units', default=[16], help='Numbers of hidden units per each attention head in each layer', type=list)
    parser.add_argument('--gat_use_residual', action='store_true', help='Whether to use residual learning in GAT')

    ### inference module related ###
    parser.add_argument('--use_enc_layer', action='store_true', help='Whether to use additional conv. layers in the inference module')
    parser.add_argument('--infer_module_loss_masking_thresh', default=0.05, help='Threshold for loss masking', type=float)
    parser.add_argument('--infer_module_kernel_size', default=3, help='Conv. kernel size for the inference module', type=int)
    parser.add_argument('--infer_module_grad_weight', default=1., help='Relative weight of the grad. on the inference module', type=float)
    parser.add_argument('--infer_module_dropout_prob', default=0.1, help='Dropout prob. for layers in the inference module', type=float)

    ### training (declared but not used) ###
    parser.add_argument('--lr', default=1e-04, help='Learning rate to use: Can be any floating point number', type=float)
    parser.add_argument('--lr_steps', default=[1000, 2000, 3000, 4000], help='When to decrease the lr during training', type=float)
    parser.add_argument('--lr_gamma', default=0.5, help='lr decay rate during training', type=float)
    parser.add_argument('--max_iters', default=5000, help='Maximum number of iterations', type=int)
    parser.add_argument('--use_graph_update', default=True, help='Whether to update graphs during training', type=bool)
    parser.add_argument('--graph_update_period', default=3000, help='Graph update period', type=int)
    parser.add_argument('--use_fov_mask', default=False, help='Whether to use fov masks', type=bool)

    args = parser.parse_args()
    return args


def make_train_qual_res(args_tuple):
    img_name, fg_prob_map, temp_graph_save_path, args = args_tuple
    if 'srns' not in args.edge_type:
        raise NotImplementedError

    win_size_str = '%.2d_%.2d'%(args.win_size, args.edge_geo_dist_thresh)
    cur_filename = os.path.basename(img_name)
    print('Regenerating a graph for ' + cur_filename + '...')
    temp = (fg_prob_map*255).astype(int)
    cur_save_path = os.path.join(temp_graph_save_path, cur_filename+'_prob.png')
    skimage.io.imsave(cur_save_path, temp)

    cur_res_graph_savepath = os.path.join(temp_graph_save_path, cur_filename+'_'+win_size_str+'.graph_res')

    # find local maxima
    vesselness = fg_prob_map

    im_y = vesselness.shape[0]
    im_x = vesselness.shape[1]
    y_quan = range(0,im_y,args.win_size)
    y_quan = sorted(list(set(y_quan) | set([im_y])))
    x_quan = range(0,im_x,args.win_size)
    x_quan = sorted(list(set(x_quan) | set([im_x])))

    max_val = []
    max_pos = []
    for y_idx in range(len(y_quan)-1):
        for x_idx in range(len(x_quan)-1):
            cur_patch = vesselness[y_quan[y_idx]:y_quan[y_idx+1],x_quan[x_idx]:x_quan[x_idx+1]]
            if np.sum(cur_patch)==0:
                max_val.append(0)
                max_pos.append((y_quan[y_idx]+cur_patch.shape[0]/2,x_quan[x_idx]+cur_patch.shape[1]/2))
            else:
                max_val.append(np.amax(cur_patch))
                temp = np.unravel_index(cur_patch.argmax(), cur_patch.shape)
                max_pos.append((y_quan[y_idx]+temp[0],x_quan[x_idx]+temp[1]))
    graph = nx.Graph()
    # add nodes
    for node_idx, (node_y, node_x) in enumerate(max_pos):
        graph.add_node(node_idx, kind='MP', y=node_y, x=node_x, label=node_idx)
        # print('node label', node_idx, 'pos', (node_y,node_x), 'added')
    speed = vesselness
    node_list = list(graph.nodes)
    for i, n in enumerate(node_list):
        phi = np.ones_like(speed)
        phi[graph.node[n]['y'],graph.node[n]['x']] = -1
        if speed[graph.node[n]['y'],graph.node[n]['x']]==0:
            continue

        neighbor = speed[max(0,graph.node[n]['y']-1):min(im_y,graph.node[n]['y']+2), \
                         max(0,graph.node[n]['x']-1):min(im_x,graph.node[n]['x']+2)]
        if np.mean(neighbor)<0.1:
            continue

        tt = skfmm.travel_time(phi, speed, narrow=args.edge_geo_dist_thresh) # travel time
        for n_comp in node_list[i+1:]:
            geo_dist = tt[graph.node[n_comp]['y'],graph.node[n_comp]['x']] # travel time
            if geo_dist < args.edge_geo_dist_thresh:
                graph.add_edge(n, n_comp, weight=args.edge_geo_dist_thresh/(args.edge_geo_dist_thresh+geo_dist))
                # print('An edge BTWN', 'node', n, '&', n_comp, 'is constructed')
    # save as files
    nx.write_gpickle(graph, cur_res_graph_savepath, protocol=pkl.HIGHEST_PROTOCOL)
    graph.clear()


def run_train(args):
    print('Called with args:')
    print(args)
    if args.use_multiprocessing:
        pool = multiprocessing.Pool(processes=args.multiprocessing_num_proc)
    train_graph_dir = f'datasets/{args.dataset}/graph'
    log_dir = os.path.join(args.save_root, args.dataset, 'VGN')
    model_save_dir = os.path.join(log_dir, 'weights')
    res_save_dir = os.path.join(log_dir, 'graph')
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    if not os.path.isdir(model_save_dir): os.mkdir(model_save_dir)
    if not os.path.isdir(res_save_dir): os.mkdir(res_save_dir)

    data_layer_train = util.GraphDataLayer(args.dataset, 'train.txt', is_training=True,
                                           edge_type=args.edge_type,
                                           win_size=args.win_size,
                                           edge_geo_dist_thresh=args.edge_geo_dist_thresh)
    data_layer_test = util.GraphDataLayer(args.dataset, 'test.txt', is_training=False,
                                          edge_type=args.edge_type,
                                          win_size=args.win_size,
                                          edge_geo_dist_thresh=args.edge_geo_dist_thresh)
    train_img_names = data_layer_train.imagepathes

    len_train = len(data_layer_train.imagepathes)
    len_test = len(data_layer_test.imagepathes)

    _, blobs_train = data_layer_train.forward()
    image_height, image_width = blobs_train['img'].shape[2:]
    network = VesselSegmVGN(args, image_width, image_height)

    if args.pretrained_model is not None:
        print("Loading model...")
        model_suffix = os.path.basename(args.pretrained_model)
        if model_suffix.endswith('pth'):
            network.load_model(args.pretrained_model)
        elif model_suffix.endswith('npy'):
            network.load_npy(args.pretrained_model)

    f_log = open(os.path.join(log_dir,'log.txt'), 'w')
    f_log.write(str(args)+'\n')
    f_log.flush()
    last_snapshot_iter = -1
    timer = util.Timer()

    # for graph update
    required_num_iters_for_train_set_update = len_train

    if args.use_graph_update:
        next_update_start = args.graph_update_period
        next_update_end = next_update_start + len_train - 1
    else:
        next_update_start = sys.maxint
        next_update_end = sys.maxint

    train_loss_list = []
    train_cnn_loss_list = []
    train_gnn_loss_list = []
    train_infer_module_loss_list = []
    test_loss_list = []
    test_cnn_loss_list = []
    test_gnn_loss_list = []
    test_infer_module_loss_list = []
    graph_update_func_arg = []
    test_loss_logs = []
    print("Training the model...")
    for iter in range(args.max_iters):
        timer.tic()
        # get one batch
        img_list, blobs_train = data_layer_train.forward()
        imgs = blobs_train['img']
        labels = blobs_train['label']

        if args.use_fov_mask:
            fov_masks = blobs_train['fov']
        else:
            fov_masks = np.ones(labels.shape, dtype=labels.dtype)

        graph = blobs_train['graph']
        num_of_nodes_list = blobs_train['num_of_nodes_list']

        node_byxs = util.get_node_byx_from_graph(graph, num_of_nodes_list)

        probmap = blobs_train['probmap']
        pixel_weights = fov_masks*((probmap>=args.infer_module_loss_masking_thresh) | labels)
        pixel_weights = pixel_weights.astype(float)

        if 'geo_dist_weighted' in args.edge_type:
            adj = nx.adjacency_matrix(graph)
        else:
            adj = nx.adjacency_matrix(graph,weight=None).astype(float)

        adj_norm = util.preprocess_graph_gat(adj)

        is_lr_flipped = False
        is_ud_flipped = False

        if blobs_train['vec_aug_on'][0]:
            is_lr_flipped = True
        if blobs_train['vec_aug_on'][1]:
            is_ud_flipped = True

        result_dict_train = network.run_batch(imgs, labels, fov_masks, node_byxs, adj_norm,
                                              pixel_weights, is_lr_flipped, is_ud_flipped,
                                              is_train=True)

        loss_val, cnn_fg_prob_mat, cnn_loss_val, cnn_accuracy_val, cnn_precision_val, cnn_recall_val, \
        gnn_loss_val, gnn_accuracy_val, infer_module_fg_prob_mat, \
        infer_module_loss_val, infer_module_accuracy_val, \
        infer_module_precision_val, infer_module_recall_val = \
        result_dict_train['loss'], result_dict_train['img_fg_prob'],\
        result_dict_train['cnn_loss'], result_dict_train['cnn_accuracy'],\
        result_dict_train['cnn_precision'], result_dict_train['cnn_recall'],\
        result_dict_train['gnn_loss'], result_dict_train['gnn_accuracy'],\
        result_dict_train['post_cnn_img_fg_prob'], result_dict_train['post_cnn_loss'],\
        result_dict_train['post_cnn_accuracy'], result_dict_train['post_cnn_precision'],\
        result_dict_train['post_cnn_recall']

        timer.toc()
        train_loss_list.append(loss_val.item())
        train_cnn_loss_list.append(cnn_loss_val.item())
        train_gnn_loss_list.append(gnn_loss_val.item())
        train_infer_module_loss_list.append(infer_module_loss_val.item())

        if (iter+1) % (cfg.DISPLAY) == 0:
            print('iter: %d / %d, loss: %.4f'%(iter+1, args.max_iters, loss_val))
            print('cnn_loss: %.4f, cnn_accuracy: %.4f, cnn_precision: %.4f, cnn_recall: %.4f'\
                    %(cnn_loss_val, cnn_accuracy_val, cnn_precision_val, cnn_recall_val))
            print('gnn_loss: %.4f, gnn_accuracy: %.4f'\
                    %(gnn_loss_val, gnn_accuracy_val))
            print('infer_module_loss: %.4f, infer_module_accuracy: %.4f, infer_module_precision: %.4f, infer_module_recall: %.4f'\
                    %(infer_module_loss_val, infer_module_accuracy_val, infer_module_precision_val, infer_module_recall_val))
            print('speed: {:.3f}s / iter'.format(timer.average_time))

        if (iter+1) % cfg.SNAPSHOT_ITERS == 0:
            last_snapshot_iter = iter
            filename = os.path.join(model_save_dir, ('iter_{:d}'.format(iter + 1) + '.pth'))
            network.save_model(filename)
            print('Wrote snapshot to: {:s}'.format(filename))

        if (iter+1) == next_update_start-1:
            data_layer_train.reinit(train_img_names, is_training=False,
                                    edge_type=args.edge_type,
                                    win_size=args.win_size,
                                    edge_geo_dist_thresh=args.edge_geo_dist_thresh)

        if ((iter+1) < args.max_iters) and ((iter+1) >= next_update_start) and ((iter+1) <= next_update_end):
            # save qualitative results
            # here, we make (segm. and corresponding) graphs,
            # which will be used as GT graphs during a next training period,
            # from current estimated vesselnesses
            cur_batch_size = len(img_list)
            infer_module_fg_prob_mat = infer_module_fg_prob_mat.detach().cpu().numpy()
            reshaped_fg_prob_map = np.reshape(infer_module_fg_prob_mat,
                                              (cur_batch_size, infer_module_fg_prob_mat.shape[1], infer_module_fg_prob_mat.shape[2]))

            for j in range(cur_batch_size):
                graph_update_func_arg.append((img_list[j], reshaped_fg_prob_map[j,:,:], train_graph_dir, args))

        if (iter+1) == next_update_end:
            if args.use_multiprocessing:
                pool.map(make_train_qual_res, graph_update_func_arg)
            else:
                for x in graph_update_func_arg:
                    make_train_qual_res(x)
            graph_update_func_arg = []

            data_layer_train.reinit(train_img_names, is_training=True,
                                    edge_type=args.edge_type,
                                    win_size=args.win_size,
                                    edge_geo_dist_thresh=args.edge_geo_dist_thresh)
            next_update_start = next_update_start + args.graph_update_period
            next_update_end = next_update_start+required_num_iters_for_train_set_update-1

        if (iter+1) % cfg.TEST_ITERS == 0:
            # cnn module related
            all_cnn_labels = np.zeros((0,))
            all_cnn_preds = np.zeros((0,))

            # gnn module related
            all_gnn_labels = np.zeros((0,))
            all_gnn_preds = np.zeros((0,))

            # inference module related
            all_infer_module_preds = np.zeros((0,))

            for _ in range(int(np.ceil(float(len_test) / cfg.GRAPH_BATCH_SIZE))):

                # get one batch
                img_list, blobs_test = data_layer_test.forward()

                imgs = blobs_test['img']
                labels = blobs_test['label']
                if args.use_fov_mask:
                    fov_masks = blobs_test['fov']
                else:
                    fov_masks = np.ones(labels.shape, dtype=labels.dtype)

                graph = blobs_test['graph']
                num_of_nodes_list = blobs_test['num_of_nodes_list']

                node_byxs = util.get_node_byx_from_graph(graph, num_of_nodes_list)
                pixel_weights = fov_masks

                if 'geo_dist_weighted' in args.edge_type:
                    adj = nx.adjacency_matrix(graph)
                else:
                    adj = nx.adjacency_matrix(graph,weight=None).astype(float)

                adj_norm = util.preprocess_graph_gat(adj)


                result_dict_test = network.run_batch(imgs, labels, fov_masks,
                                                     node_byxs=node_byxs,
                                                     adj=adj_norm,
                                                     pixel_weights=pixel_weights,
                                                     is_lr_flipped=False,
                                                     is_ud_flipped=False,
                                                     is_train=False)

                loss_val, cnn_fg_prob_mat, cnn_loss_val, \
                gnn_labels, gnn_prob_vec, gnn_loss_val, \
                infer_module_fg_prob_mat, infer_module_loss_val = \
                result_dict_test['loss'], result_dict_test['img_fg_prob'], \
                result_dict_test['cnn_loss'], result_dict_test['node_labels'], \
                result_dict_test['gnn_prob'], result_dict_test['gnn_loss'], \
                result_dict_test['post_cnn_img_fg_prob'], result_dict_test['post_cnn_loss']


                gnn_labels = gnn_labels.detach().cpu().numpy()
                gnn_prob_vec = gnn_prob_vec.detach().cpu().numpy()
                cnn_fg_prob_mat = cnn_fg_prob_mat.detach().cpu().numpy()
                infer_module_fg_prob_mat = infer_module_fg_prob_mat.detach().cpu().numpy()

                cnn_fg_prob_mat = cnn_fg_prob_mat*fov_masks.astype(float)
                infer_module_fg_prob_mat = infer_module_fg_prob_mat*fov_masks.astype(float)

                test_loss_list.append(loss_val.item())
                test_cnn_loss_list.append(cnn_loss_val.item())
                test_gnn_loss_list.append(gnn_loss_val.item())
                test_infer_module_loss_list.append(infer_module_loss_val.item())

                all_cnn_labels = np.concatenate((all_cnn_labels,np.reshape(labels, (-1))))
                all_cnn_preds = np.concatenate((all_cnn_preds,np.reshape(cnn_fg_prob_mat, (-1))))

                all_gnn_labels = np.concatenate((all_gnn_labels,gnn_labels))
                all_gnn_preds = np.concatenate((all_gnn_preds,gnn_prob_vec))

                all_infer_module_preds = np.concatenate((all_infer_module_preds,np.reshape(infer_module_fg_prob_mat, (-1))))

                # save qualitative results
                cur_batch_size = len(img_list)
                reshaped_cnn_fg_prob_map = cnn_fg_prob_mat.reshape((cur_batch_size,cnn_fg_prob_mat.shape[1],cnn_fg_prob_mat.shape[2]))
                reshaped_infer_module_fg_prob_mat = infer_module_fg_prob_mat.reshape((cur_batch_size,infer_module_fg_prob_mat.shape[1],infer_module_fg_prob_mat.shape[2]))
                for j in range(cur_batch_size):
                    cur_img_name = os.path.basename(img_list[j])

                    cur_cnn_fg_prob_map = reshaped_cnn_fg_prob_map[j,:,:]
                    cur_infer_module_fg_prob_map = reshaped_infer_module_fg_prob_mat[j,:,:]

                    cur_map = (cur_cnn_fg_prob_map*255).astype(int)
                    cur_save_path = os.path.join(res_save_dir, cur_img_name + '_prob_cnn.png')
                    skimage.io.imsave(cur_save_path, cur_map)
                    cur_map = (cur_infer_module_fg_prob_map*255).astype(int)
                    # cur_map[cur_map==127] = 0
                    cur_save_path = os.path.join(res_save_dir, cur_img_name + '_prob_infer_module.png')
                    skimage.io.imsave(cur_save_path, cur_map)

            cnn_auc_test, cnn_ap_test = util.get_auc_ap_score(all_cnn_labels, all_cnn_preds)
            all_cnn_labels_bin = np.copy(all_cnn_labels).astype(np.bool)
            all_cnn_preds_bin = all_cnn_preds>=0.5
            all_cnn_correct = all_cnn_labels_bin==all_cnn_preds_bin
            cnn_acc_test = np.mean(all_cnn_correct.astype(np.float32))

            gnn_auc_test, gnn_ap_test = util.get_auc_ap_score(all_gnn_labels, all_gnn_preds)
            all_gnn_labels_bin = np.copy(all_gnn_labels).astype(np.bool)
            all_gnn_preds_bin = all_gnn_preds>=0.5
            all_gnn_correct = all_gnn_labels_bin==all_gnn_preds_bin
            gnn_acc_test = np.mean(all_gnn_correct.astype(np.float32))

            infer_module_auc_test, infer_module_ap_test = util.get_auc_ap_score(all_cnn_labels, all_infer_module_preds)
            all_infer_module_preds_bin = all_infer_module_preds>=0.5
            all_infer_module_correct = all_cnn_labels_bin==all_infer_module_preds_bin
            infer_module_acc_test = np.mean(all_infer_module_correct.astype(np.float32))

            print('iter: %d / %d, train_loss: %.4f, train_cnn_loss: %.4f, train_gnn_loss: %.4f, train_infer_module_loss: %.4f'\
                %(iter+1, args.max_iters, np.mean(train_loss_list), np.mean(train_cnn_loss_list), np.mean(train_gnn_loss_list), np.mean(train_infer_module_loss_list)))
            print('iter: %d / %d, test_loss: %.4f, test_cnn_loss: %.4f, test_gnn_loss: %.4f, test_infer_module_loss: %.4f'\
                %(iter+1, args.max_iters, np.mean(test_loss_list), np.mean(test_cnn_loss_list), np.mean(test_gnn_loss_list), np.mean(test_infer_module_loss_list)))
            print('test_cnn_acc: %.4f, test_cnn_auc: %.4f, test_cnn_ap: %.4f'%(cnn_acc_test, cnn_auc_test, cnn_ap_test))
            print('test_gnn_acc: %.4f, test_gnn_auc: %.4f, test_gnn_ap: %.4f'%(gnn_acc_test, gnn_auc_test, gnn_ap_test))
            print('test_infer_module_acc: %.4f, test_infer_module_auc: %.4f, test_infer_module_ap: %.4f'%(infer_module_acc_test, infer_module_auc_test, infer_module_ap_test))


            f_log.write('iter: '+str(iter+1)+' / '+str(args.max_iters)+'\n')
            f_log.write('train_loss '+str(np.mean(train_loss_list))+'\n')
            f_log.write('train_cnn_loss '+str(np.mean(train_cnn_loss_list))+'\n')
            f_log.write('train_gnn_loss '+str(np.mean(train_gnn_loss_list))+'\n')
            f_log.write('train_infer_module_loss '+str(np.mean(train_infer_module_loss_list))+'\n')
            f_log.write('iter: '+str(iter+1)+' / '+str(args.max_iters)+'\n')
            f_log.write('test_loss '+str(np.mean(test_loss_list))+'\n')
            f_log.write('test_cnn_loss '+str(np.mean(test_cnn_loss_list))+'\n')
            f_log.write('test_gnn_loss '+str(np.mean(test_gnn_loss_list))+'\n')
            f_log.write('test_infer_module_loss '+str(np.mean(test_infer_module_loss_list))+'\n')
            f_log.write('test_cnn_acc '+str(cnn_acc_test)+'\n')
            f_log.write('test_cnn_auc '+str(cnn_auc_test)+'\n')
            f_log.write('test_cnn_ap '+str(cnn_ap_test)+'\n')
            f_log.write('test_gnn_acc '+str(gnn_acc_test)+'\n')
            f_log.write('test_gnn_auc '+str(gnn_auc_test)+'\n')
            f_log.write('test_gnn_ap '+str(gnn_ap_test)+'\n')
            f_log.write('test_infer_module_acc '+str(infer_module_acc_test)+'\n')
            f_log.write('test_infer_module_auc '+str(infer_module_auc_test)+'\n')
            f_log.write('test_infer_module_ap '+str(infer_module_ap_test)+'\n')

            f_log.flush()

            test_loss_logs.append(float(np.mean(test_loss_list)))

            train_loss_list = []
            train_cnn_loss_list = []
            train_gnn_loss_list = []
            train_infer_module_loss_list = []
            test_loss_list = []
            test_cnn_loss_list = []
            test_gnn_loss_list = []
            test_infer_module_loss_list = []

            # cnn module related
            all_cnn_labels = np.zeros((0,))
            all_cnn_preds = np.zeros((0,))

            # gnn module related
            all_gnn_labels = np.zeros((0,))
            all_gnn_preds = np.zeros((0,))

            # inference module related
            all_infer_module_preds = np.zeros((0,))

    if last_snapshot_iter != iter:
        filename = os.path.join(model_save_dir,('iter_{:d}'.format(iter+1) + '.pth'))
        network.save_model(filename)
        print('Wrote snapshot to: {:s}'.format(filename))

    f_log.close()
    print("Training complete.")


if __name__ == '__main__':
    args = parse_args()
    run_train(args)
