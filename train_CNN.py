import numpy as np
import os
import argparse
import cv2

import config as cfg
from models.cnn import VesselSegmCNN
import util


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a vessel_segm_cnn network')
    parser.add_argument('--dataset', default='HRF', help='Dataset to use: Can be DRIVE or STARE or CHASE_DB1 or HRF', type=str)
    parser.add_argument('--cnn_model', default='driu', help='CNN model to use', type=str)
    parser.add_argument('--lr', default=1e-02, help='Learning rate to use: Can be any floating point number', type=float)
    parser.add_argument('--max_iters', default=5000, help='Maximum number of iterations', type=int)
    parser.add_argument('--pretrained_model', default='pretrained_model/VGG_imagenet.npy', help='path for a pretrained model(.pth)', type=str)
    parser.add_argument('--save_root', default='log', help='root path to save trained models and test results', type=str)
    args = parser.parse_args()
    return args




def run_train(args):
    print('Called with args:')
    print(args)

    data_layer_train = util.DataLayer(args.dataset, is_training=True)
    data_layer_test = util.DataLayer(args.dataset, is_training=False)

    log_dir = os.path.join(args.save_root, args.dataset)
    model_save_dir = os.path.join(log_dir, 'weights')
    res_save_dir = os.path.join(log_dir, 'graph')

    if not os.path.exists(log_dir): os.mkdir(log_dir)
    if not os.path.isdir(model_save_dir): os.mkdir(model_save_dir)
    if not os.path.isdir(res_save_dir): os.mkdir(res_save_dir)

    network = VesselSegmCNN(args)

    if args.pretrained_model is not None:
        print("Loading pretrained model...")
        network.load_npy(args.pretrained_model)

    f_log = open(os.path.join(log_dir,'log.txt'), 'w')
    last_snapshot_iter = -1
    timer = util.Timer()

    train_loss_list = []
    test_loss_list = []
    print("Training the model...")
    for iter in range(args.max_iters):
        timer.tic()
        # get one batch
        _, blobs_train = data_layer_train.forward()
        fov_masks = np.ones(blobs_train['label'].shape, dtype=blobs_train['label'].dtype)

        loss_val, accuracy_val, pre_val, rec_val, _ = network.run_batch(blobs_train['img'], blobs_train['label'], fov_masks)
        timer.toc()
        train_loss_list.append(loss_val.item())

        if (iter+1) % (cfg.DISPLAY) == 0:
            print('iter: %d / %d, loss: %.4f, accuracy: %.4f, precision: %.4f, recall: %.4f'\
                    %(iter+1, args.max_iters, loss_val, accuracy_val, pre_val, rec_val))
            print('speed: {:.3f}s / iter'.format(timer.average_time))

        if (iter+1) % cfg.SNAPSHOT_ITERS == 0:
            last_snapshot_iter = iter
            filename = os.path.join(model_save_dir, ('iter_{:d}'.format(iter+1) + '.pth'))
            network.save_model(filename)
            print('Wrote snapshot to: {:s}'.format(filename))

        if (iter+1) % cfg.TEST_ITERS == 0:
            all_labels = np.zeros((0,))
            all_preds = np.zeros((0,))

            for _ in range(int(np.ceil(float(len(data_layer_test.imagepathes)) / cfg.BATCH_SIZE))):
                # get one batch
                img_list, blobs_test = data_layer_test.forward()

                imgs = blobs_test['img']
                labels = blobs_test['label']
                fov_masks = np.ones(labels.shape, dtype=labels.dtype)
                loss_val, *_, fg_prob_map = network.run_batch(imgs, labels, fov_masks, False)

                test_loss_list.append(loss_val.item())
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


            print('iter: %d / %d, train_loss: %.4f'%(iter+1, args.max_iters, np.mean(train_loss_list)))
            print('iter: %d / %d, test_loss: %.4f, test_acc: %.4f, test_auc: %.4f, test_ap: %.4f'\
                  %(iter+1, args.max_iters, np.mean(test_loss_list), acc_test, auc_test, ap_test))

            f_log.write('iter: ' + str(iter+1) + ' / '+ str(args.max_iters) + '\n')
            f_log.write('train_loss ' + str(np.mean(train_loss_list)) + '\n')
            f_log.write('iter: ' + str(iter+1) + ' / '+ str(args.max_iters) + '\n')
            f_log.write('test_loss ' + str(np.mean(test_loss_list)) + '\n')
            f_log.write('test_acc ' + str(acc_test) + '\n')
            f_log.write('test_auc ' + str(auc_test) + '\n')
            f_log.write('test_ap ' + str(ap_test) + '\n')
            f_log.flush()

            train_loss_list = []
            test_loss_list = []

    if last_snapshot_iter != iter:
        filename = os.path.join(model_save_dir, ('iter_{:d}'.format(iter+1) + '.pth'))
        network.save_model(filename)
        print('Wrote snapshot to: {:s}'.format(filename))

    f_log.close()
    print("Training complete.")

if __name__ == '__main__':
    args = parse_args()
    run_train(args)



