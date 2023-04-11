import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
import copy


import config as cfg
from module import DRIU, LargeDRIU



class VesselSegmCNN():
    def __init__(self, params):
        self.params = params
        if params.cnn_model == 'driu':
            self.build_driu()
        elif params.cnn_model == 'driu_large':
            self.build_driu_large()
        else:
            raise ValueError('Invalid cnn_model params!')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cnn_model = self.cnn_model.to(self.device)
        print(f"Model built on {self.device}.")
        opt_params = {
            'params': filter(lambda p: p.requires_grad, self.cnn_model.parameters()),
            'lr': params.lr,
            'weight_decay': cfg.WEIGHT_DECAY_RATE
            }
        self.optimizer = opt.Adam(**opt_params)
        self.loss_function = nn.BCEWithLogitsLoss(reduce=False)

    def build_driu(self):
        self.cnn_model = DRIU()

    def build_driu_large(self):
        self.cnn_model = LargeDRIU()

    def load_model(self, modelpath):
        self.cnn_model.load_state_dict(torch.load(modelpath))

    def load_npy(self, data_path):
        data_dict = np.load(data_path, allow_pickle=True, encoding='latin1').item()
        state_dict = copy.deepcopy(self.cnn_model.state_dict())
        assgned_keys = []
        for key in state_dict:
            layer_name = key.split('.')[0]
            if layer_name in data_dict:
                if layer_name in assgned_keys: continue
                weight = data_dict[layer_name]['weights']
                bias = data_dict[layer_name]['biases']
                weight_key = f'{layer_name}.0.weight'
                bias_key = f'{layer_name}.0.bias'
                weight = np.transpose(weight, (3, 2, 0, 1))
                state_dict[weight_key] = torch.from_numpy(weight).to(self.device)
                state_dict[bias_key] = torch.from_numpy(bias).to(self.device)
                print("assign pretrain model " + layer_name + "_weights to " + weight_key)
                print("assign pretrain model " + layer_name + "_biases to " + bias_key)
                assgned_keys.append(layer_name)
        self.cnn_model.load_state_dict(state_dict)

    def save_model(self, modelpath):
        torch.save(self.cnn_model.state_dict(), modelpath)

    def forward(self, imgs, is_train=False):
        input_imgs = torch.from_numpy(imgs).to(self.device)
        if is_train: return self.cnn_model(input_imgs)
        with torch.no_grad():
            return self.cnn_model(input_imgs)

    def run_batch(self, imgs, labels, fov_masks, is_train=True):
        if is_train:
            self.optimizer.zero_grad(set_to_none=True)
        output = self.forward(imgs, is_train).permute([0, 2, 3, 1])
        labels = torch.from_numpy(labels).to(self.device).long()
        fov_masks = torch.from_numpy(fov_masks).to(self.device).long()
        fg_prob = torch.sigmoid(output)

        ### Compute the loss ###
        binary_mask_fg = labels == 1
        binary_mask_bg = labels != 1
        combined_mask = torch.cat([binary_mask_bg, binary_mask_fg], dim=3).float()
        flat_one_hot_labels = combined_mask.view(-1, 2)

        flat_labels = labels.view(-1,).float()
        flat_logits = output.view(-1,)
        cross_entropies = self.loss_function(flat_logits, flat_labels)

        # weighted cross entropy loss (in fov)
        num_pixel = torch.sum(fov_masks)
        num_pixel_fg = torch.count_nonzero(binary_mask_fg)
        num_pixel_bg = num_pixel - num_pixel_fg

        class_weight = torch.cat([torch.reshape(num_pixel_fg / num_pixel, (1, 1)),
                                  torch.reshape(num_pixel_bg / num_pixel, (1, 1))], axis=1).float()
        weight_per_label = torch.matmul(flat_one_hot_labels, class_weight.permute([1, 0])).permute([1, 0]) #shape [1, TRAIN.BATCH_SIZE]

        # this is the weight for each datapoint, depending on its label
        reshaped_fov_masks = fov_masks.view(-1,).float()
        reshaped_fov_masks /= torch.mean(reshaped_fov_masks)

        loss = torch.mean(torch.mul(torch.mul(reshaped_fov_masks, weight_per_label), cross_entropies))

        if is_train:
            loss.backward()
            self.optimizer.step()

        ### Compute the accuracy ###
        flat_bin_output = fg_prob.view((-1,)) >= 0.5
        # accuracy
        correct = torch.eq(flat_bin_output, flat_labels.bool()).float()
        accuracy = torch.mean(correct)
        # precision, recall
        num_fg_output = torch.sum(flat_bin_output.float())
        tp = torch.sum(torch.logical_and(flat_labels.bool(), flat_bin_output).float())
        precision = torch.divide(tp, torch.add(num_fg_output, cfg.EPSILON))
        recall = torch.divide(tp, num_pixel_fg.float())
        return loss, accuracy, precision, recall, fg_prob.detach().cpu().numpy()


