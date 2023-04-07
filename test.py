from model import VesselSegmCNN
import argparse
import torch
import util
import cv2

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a vessel_segm_cnn network')
    parser.add_argument('--dataset', default='CHASE_DB1', help='Dataset to use: Can be CHASE_DB1 or HRF', type=str)
    parser.add_argument('--cnn_model', default='driu', help='CNN model to use', type=str)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)


    # network = VesselSegmCNN(args)
    # # print(network.cnn_model)
    # x = torch.rand((1, 3, 768, 768))
    # print(x.size())
    # print(network.cnn_model(x).size())


    import numpy as np
    import matplotlib.pyplot as plt
    data_layer_train = util.DataLayer(args.dataset, is_training=True, use_padding=True)
    img_list, blobs = data_layer_train.forward()
    print(img_list)
    print(blobs['img'].shape)
    image = blobs['img'][0]
    image = np.transpose(image, axes=[1, 2, 0])
    print(image[0])
    plt.imshow(image)
    plt.show()
