import argparse
import cv2
import os
import matplotlib.pyplot as plt


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--dataset', default='Artery', help='Dataset to use: Can be Artery or HRF', type=str)

    return parser.parse_args()


def draw_cnn_results(args):
    result_dir = f'log/{args.dataset}'
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


if __name__ == '__main__':
    args = parse_args()
    draw_cnn_results(args)






