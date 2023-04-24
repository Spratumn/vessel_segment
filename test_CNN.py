import argparse
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import config as cfg
from models.cnn import VesselSegmCNN
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
    result_dir = f'log/{args.dataset}/CNN'
    log_path = os.path.join(result_dir, 'log.txt')
    with open(log_path, 'r') as f:
        lines = f.readlines()
    logs = [l.rstrip('\n') for l in lines]
    iter_idxes = []
    train_losses = []
    test_losses = []
    test_accs = []
    test_aucs = []
    test_aps = []
    for i in range(0, len(logs), 7):
        iter_idxes.append(int(logs[i].split(' ')[1]))
        train_losses.append(float(logs[i+1].split(' ')[1]))
        test_losses.append(float(logs[i+3].split(' ')[1]))
        test_accs.append(float(logs[i+4].split(' ')[1]))
        test_aucs.append(float(logs[i+5].split(' ')[1]))
        test_aps.append(float(logs[i+6].split(' ')[1]))

    plt.plot(iter_idxes, train_losses, c='blue', label='train')
    plt.plot(iter_idxes, test_losses, c='green', label='test')
    plt.legend()
    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.savefig(os.path.join(result_dir, 'loss.png'))
    plt.close()

    plt.plot(iter_idxes, test_accs, c='blue', label='acc')
    plt.plot(iter_idxes, test_aucs, c='green', label='auc')
    plt.plot(iter_idxes, test_aps, c='red', label='ap')
    plt.legend()
    plt.xlabel('iters')
    plt.ylabel('performance')
    plt.savefig(os.path.join(result_dir, 'performance.png'))
    plt.close()



def get_prob(args, ds_filename='test.txt', modelpath='', res_save_dir=''):
    network = VesselSegmCNN(args)
    network.load_model(modelpath)
    data_layer = util.DataLayer(args.dataset, ds_filename, is_training=False)

    all_labels = np.zeros((0,))
    all_preds = np.zeros((0,))

    for _ in tqdm(range(int(np.ceil(float(len(data_layer.imagepathes)) / cfg.BATCH_SIZE)))):
        # get one batch
        img_list, blobs_test = data_layer.forward()

        imgs = blobs_test['img']
        labels = blobs_test['label']
        fov_masks = np.ones(labels.shape, dtype=labels.dtype)
        *_, fg_prob_map = network.run_batch(imgs, labels, fov_masks, False)

        all_labels = np.concatenate((all_labels, np.reshape(labels, (-1))))
        fg_prob_map = fg_prob_map * fov_masks.astype(float)
        all_preds = np.concatenate((all_preds, np.reshape(fg_prob_map, (-1))))
        # save qualitative results
        cur_batch_size = len(img_list)
        reshaped_fg_prob_map = fg_prob_map.reshape((cur_batch_size, fg_prob_map.shape[1],fg_prob_map.shape[2]))
        reshaped_output = reshaped_fg_prob_map >= 0.5
        for img_idx in range(cur_batch_size):
            cur_test_img_path = img_list[img_idx]
            temp_name = os.path.basename(cur_test_img_path)

            cur_reshaped_fg_prob_map = (reshaped_fg_prob_map[img_idx,:,:] * 255).astype(int)
            cur_reshaped_output = reshaped_output[img_idx,:,:].astype(int) * 255

            cur_fg_prob_save_path = os.path.join(res_save_dir, temp_name + '_prob.png')
            cur_output_save_path = os.path.join(res_save_dir, temp_name + '_output.png')

            cv2.imwrite(cur_fg_prob_save_path, cur_reshaped_fg_prob_map)
            cv2.imwrite(cur_output_save_path, cur_reshaped_output)

    auc_test, ap_test = util.get_auc_ap_score(all_labels, all_preds)
    all_labels_bin = np.copy(all_labels).astype(np.bool)
    all_preds_bin = all_preds >= 0.5
    all_correct = all_labels_bin == all_preds_bin
    acc_test = np.mean(all_correct.astype(np.float32))

    print('acc: %.4f, auc: %.4f, ap: %.4f'%(acc_test, auc_test, ap_test))



if __name__ == '__main__':
    args = parse_args()
    # 绘制训练相关曲线图
    draw_cnn_results(args)

    # 生成prob图片
    # 训练集
    get_prob(args, ds_filename='train.txt',
             modelpath='log/Artery/CNN/weights/iter_5000.pth',
             res_save_dir='datasets/Artery/graph')
    # 测试集
    get_prob(args, ds_filename='test.txt',
             modelpath='log/Artery/CNN/weights/iter_5000.pth',
             res_save_dir='datasets/Artery/graph')







