import argparse
import cv2
import os
import skimage.io
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import config as cfg
from models.vgn import VesselSegmVGN
import util

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--dataset', default='Artery', help='Dataset to use: Can be Artery or HRF', type=str)
    parser.add_argument('--cnn_model', default='driu', help='CNN model to use', type=str)

    return parser.parse_args()


def draw_cnn_results(args):
    result_dir = f'log/{args.dataset}'
    log_path = os.path.join(result_dir, 'log.txt')
    with open(log_path, 'r') as f:
        lines = f.readlines()
    logs = [l.rstrip('\n') for l in lines[1:]]
    iter_idxes = []
    train_losses = []
    train_cnn_losses = []
    train_gnn_losses = []
    train_infer_module_loss = []
    test_losses = []
    test_cnn_losses = []
    test_gnn_losses = []
    test_infer_module_losses = []
    test_cnn_accs = []
    test_cnn_aucs = []
    test_cnn_aps = []
    test_gnn_accs = []
    test_gnn_aucs = []
    test_gnn_aps = []
    test_infer_module_accs = []
    test_infer_module_aucs = []
    test_infer_module_aps = []

    for i in range(0, len(logs), 19):
        iter_idxes.append(int(logs[i].split(' ')[1]))
        train_losses.append(float(logs[i+1].split(' ')[1]))
        train_cnn_losses.append(float(logs[i+2].split(' ')[1]))
        train_gnn_losses.append(float(logs[i+3].split(' ')[1]))
        train_infer_module_loss.append(float(logs[i+4].split(' ')[1]))

        test_losses.append(float(logs[i+6].split(' ')[1]))
        test_cnn_losses.append(float(logs[i+7].split(' ')[1]))
        test_gnn_losses.append(float(logs[i+8].split(' ')[1]))
        test_infer_module_losses.append(float(logs[i+9].split(' ')[1]))

        test_cnn_accs.append(float(logs[i+10].split(' ')[1]))
        test_cnn_aucs.append(float(logs[i+11].split(' ')[1]))
        test_cnn_aps.append(float(logs[i+12].split(' ')[1]))
        test_gnn_accs.append(float(logs[i+13].split(' ')[1]))
        test_gnn_aucs.append(float(logs[i+14].split(' ')[1]))
        test_gnn_aps.append(float(logs[i+15].split(' ')[1]))
        test_infer_module_accs.append(float(logs[i+16].split(' ')[1]))
        test_infer_module_aucs.append(float(logs[i+17].split(' ')[1]))
        test_infer_module_aps.append(float(logs[i+18].split(' ')[1]))

    plt.plot(iter_idxes, train_losses, label='train')
    plt.plot(iter_idxes, test_losses, label='test')
    plt.plot(iter_idxes, train_cnn_losses, label='cnn')
    plt.plot(iter_idxes, train_gnn_losses, label='gnn')
    plt.plot(iter_idxes, train_infer_module_loss, label='infer')
    plt.legend()
    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.savefig(os.path.join(result_dir, 'loss.png'))
    plt.close()

    plt.plot(iter_idxes, test_cnn_accs, label='cnn_acc')
    plt.plot(iter_idxes, test_cnn_aucs, label='cnn_auc')
    plt.plot(iter_idxes, test_cnn_aps, label='cnn_ap')
    plt.plot(iter_idxes, test_gnn_accs, label='gnn_acc')
    plt.plot(iter_idxes, test_gnn_aucs, label='gnn_auc')
    plt.plot(iter_idxes, test_gnn_aps, label='gnn_ap')
    plt.plot(iter_idxes, test_infer_module_accs, label='infer_acc')
    plt.plot(iter_idxes, test_infer_module_aucs, label='infer_auc')
    plt.plot(iter_idxes, test_infer_module_aps, label='infer_ap')
    plt.legend()
    plt.xlabel('iters')
    plt.ylabel('performance')
    plt.savefig(os.path.join(result_dir, 'performance.png'))
    plt.close()



def get_prob(args, ds_filename='test.txt', modelpath='', res_save_dir=''):
    network = VesselSegmVGN(args)
    network.load_model(modelpath)
    data_layer = util.GraphDataLayer(args.dataset, ds_filename, is_training=False)

    # cnn module related
    all_cnn_labels = np.zeros((0,))
    all_cnn_preds = np.zeros((0,))

    # gnn module related
    all_gnn_labels = np.zeros((0,))
    all_gnn_preds = np.zeros((0,))

    # inference module related
    all_infer_module_preds = np.zeros((0,))

    for _ in tqdm(range(int(np.ceil(float(len(data_layer.imagepathes)) / cfg.GRAPH_BATCH_SIZE)))):
        # get one batch
        img_list, blobs_test = data_layer.forward()
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

        cnn_fg_prob_mat, \
        gnn_labels, gnn_prob_vec, \
        infer_module_fg_prob_mat = \
        result_dict_test['img_fg_prob'], \
        result_dict_test['node_labels'], \
        result_dict_test['gnn_prob'], \
        result_dict_test['post_cnn_img_fg_prob']


        gnn_labels = gnn_labels.detach().cpu().numpy()
        gnn_prob_vec = gnn_prob_vec.detach().cpu().numpy()
        cnn_fg_prob_mat = cnn_fg_prob_mat.detach().cpu().numpy()
        infer_module_fg_prob_mat = infer_module_fg_prob_mat.detach().cpu().numpy()

        cnn_fg_prob_mat = cnn_fg_prob_mat*fov_masks.astype(float)
        infer_module_fg_prob_mat = infer_module_fg_prob_mat*fov_masks.astype(float)

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

    print('test_cnn_acc: %.4f, test_cnn_auc: %.4f, test_cnn_ap: %.4f'%(cnn_acc_test, cnn_auc_test, cnn_ap_test))
    print('test_gnn_acc: %.4f, test_gnn_auc: %.4f, test_gnn_ap: %.4f'%(gnn_acc_test, gnn_auc_test, gnn_ap_test))
    print('test_infer_module_acc: %.4f, test_infer_module_auc: %.4f, test_infer_module_ap: %.4f'%(infer_module_acc_test, infer_module_auc_test, infer_module_ap_test))






if __name__ == '__main__':
    args = parse_args()
    # 绘制训练相关曲线图
    draw_cnn_results(args)

    # 生成prob图片
    # 训练集
    get_prob(args, ds_filename='train.txt',
             modelpath='log/Artery/VGN/weights/iter_5000.pth',
             res_save_dir='datasets/Artery/graph')
    # 测试集
    # get_prob(args, ds_filename='test.txt',
    #          modelpath='log/Artery/CNN/weights/iter_5000.pth',
    #          res_save_dir='datasets/Artery/graph')







