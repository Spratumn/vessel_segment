import torch
import torch.nn as nn
import torch.optim as opt
import itertools
import numpy as np
import copy

import config as cfg
from .module import DRIU, LargeDRIU, new_conv_layer, new_deconv_layer


class VesselSegmVGN():
    def __init__(self, params):
        self.params = params
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.cnn_model =self.build_cnn(params.cnn_model)
        self.gat_model = GAT(params)
        self.infer_model = Infer(params)
        self.cnn_loss_function = nn.BCEWithLogitsLoss(reduce=False)
        self.post_cnn_loss_function = nn.BCEWithLogitsLoss(reduce=False)
        self.gnn_loss_function = nn.BCEWithLogitsLoss(reduce=False)

        opt_params = {
            'params': filter(lambda p: p.requires_grad,
                             itertools.chain(self.cnn_model.parameters(),
                                             self.gat_model.parameters(),
                                             self.infer_model.parameters())),
            'lr': params.lr,
            'weight_decay': cfg.WEIGHT_DECAY_RATE
            }
        self.optimizer = opt.Adam(**opt_params)

    def build_cnn(self, cnn_model='driu'):
        if cnn_model == 'driu':
            return DRIU(for_vgn=True)
        elif cnn_model == 'driu_large':
            return LargeDRIU(for_vgn=True)
        else:
            raise ValueError('Invalid cnn_model params!')

    def save_model(self, modelpath):
        weight_dict = {
            'cnn': self.cnn_model.state_dict(),
            'gat': self.gat_model.state_dict(),
            'infer': self.infer_model.state_dict(),
        }
        torch.save(weight_dict, modelpath)

    def load_model(self, modelpath):
        weight_dict = torch.load(modelpath)
        self.cnn_model.load_state_dict(weight_dict['cnn'])
        self.gat_model.load_state_dict(weight_dict['gat'])
        self.infer_model.load_state_dict(weight_dict['infer'])

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

    def load_cnn(self, cnn_model_path):
        cnn_state_dict = torch.load(cnn_model_path)
        self.cnn_model.load_state_dict(cnn_state_dict)
        print('Load CNN model Succeed!')

    def forward(self, imgs, labels, fov_masks, node_byxs, adj,
                is_lr_flipped=False, is_ud_flipped=False, is_train=False):
        input_imgs = torch.from_numpy(imgs).to(self.device)
        labels = torch.from_numpy(labels).to(self.device).long()
        fov_masks = torch.from_numpy(fov_masks).to(self.device).long()
        if is_train:
            cnn_feat, conv_feats, img_output = self.cnn_model(input_imgs)
            node_logits, gnn_final_feats, node_byxs, node_labels = self.gat_model(node_byxs, adj, conv_feats, labels)
            post_cnn_img_fg_prob, post_cnn_img_output = self.infer_model(imgs, cnn_feat, gnn_final_feats,
                                                                         is_lr_flipped, is_ud_flipped)
        with torch.no_grad():
            cnn_feat, conv_feats, img_output = self.cnn_model(input_imgs)
            node_logits, gnn_final_feats, node_byxs, node_labels = self.gat_model(node_byxs, adj, conv_feats, labels)
            post_cnn_img_fg_prob, post_cnn_img_output = self.infer_model(imgs, cnn_feat, gnn_final_feats,
                                                                         is_lr_flipped, is_ud_flipped)

        return {'post_cnn_img_fg_prob': post_cnn_img_fg_prob,
                'img_output': img_output,
                'post_cnn_img_output': post_cnn_img_output,
                'node_logits': node_logits,
                'node_labels': node_labels}

    def run_batch(self, imgs, labels, fov_masks, node_byxs, adj,
                  pixel_weights, is_lr_flipped, is_ud_flipped,
                  is_train=True):

        forward_dict = self.forward(imgs, labels, fov_masks, node_byxs, adj,
                                    is_lr_flipped, is_ud_flipped, is_train)
        post_cnn_img_fg_prob = forward_dict['post_cnn_img_fg_prob']
        img_output = forward_dict['img_output']
        post_cnn_img_output = forward_dict['post_cnn_img_output']
        node_logits = forward_dict['node_logits']
        node_labels = forward_dict['node_labels']

        img_fg_prob = torch.sigmoid(img_output)
        ### Compute the loss ###
        binary_mask_fg = labels == 1
        binary_mask_bg = labels != 1
        combined_mask = torch.cat([binary_mask_bg, binary_mask_fg], dim=3).float()
        flat_one_hot_labels = combined_mask.view(-1, 2)
        flat_labels = labels.view(-1,).float()
        flat_logits = img_output.view(-1,)
        cnn_cross_entropies = self.cnn_loss_function(flat_logits, flat_labels)

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
        cnn_loss = torch.mean(torch.mul(torch.mul(reshaped_fov_masks, weight_per_label), cnn_cross_entropies))

        ### Compute the accuracy ###
        flat_bin_output = img_fg_prob.view((-1,)) >= 0.5
        # accuracy
        cnn_correct = torch.eq(flat_bin_output, flat_labels.bool()).float()
        cnn_accuracy = torch.mean(cnn_correct)
        # precision, recall
        num_fg_output = torch.sum(flat_bin_output.float())
        cnn_tp = torch.sum(torch.logical_and(flat_labels.bool(), flat_bin_output).float())
        cnn_precision = torch.divide(cnn_tp, torch.add(num_fg_output, cfg.EPSILON))
        cnn_recall = torch.divide(cnn_tp, num_pixel_fg.float())

        ###### gnn related ######
        ### Compute the loss ###
        gnn_cross_entropies = self.gnn_loss_function(node_logits, node_labels)

        # simple cross entropy loss
        # weighted cross entropy loss
        num_node = 1
        for n in node_labels.size(): num_node *= n

        num_node_fg = torch.count_nonzero(node_labels)
        num_node_bg = num_node - num_node_fg
        gnn_class_weight = torch.cat([torch.reshape(num_node_fg / num_node,(1,1)),
                                      torch.reshape(num_node_bg / num_node,(1,1))], axis=1).float()
        gnn_weight_per_label = torch.matmul(torch.nn.functional.one_hot(node_labels.float(), 2), gnn_class_weight.permute([1, 0])).permute([1, 0])
        # this is the weight for each datapoint, depending on its label
        gnn_loss = torch.mean(torch.mul(gnn_weight_per_label,gnn_cross_entropies))

        ### Compute the accuracy ###
        gnn_prob = torch.sigmoid(node_logits)
        gnn_correct = torch.eq((gnn_prob >= 0.5).long(), node_labels.long())
        gnn_accuracy = torch.mean(gnn_correct.float())

        ###### inference module related ######
        ### Compute the loss ###
        post_cnn_flat_logits = torch.reshape(post_cnn_img_output, shape=(-1,))
        post_cnn_cross_entropies = self.post_cnn_loss_function(post_cnn_flat_logits, flat_labels.float())

        # weighted cross entropy loss
        reshaped_pixel_weights = torch.reshape(pixel_weights, shape=(-1,))
        reshaped_pixel_weights /= torch.mean(reshaped_pixel_weights)
        post_cnn_loss = torch.mean(torch.mul(torch.mul(reshaped_pixel_weights, weight_per_label), post_cnn_cross_entropies))

        loss = post_cnn_loss
        if self.params.cnn_loss_on:
            loss += cnn_loss

        if self.params.gnn_loss_on:
            loss += gnn_loss * self.params.gnn_loss_weight

        if is_train:
            loss.backward()
            self.optimizer.step()

        ### Compute the accuracy ###
        post_cnn_flat_bin_output = torch.reshape(post_cnn_img_fg_prob, shape=(-1,)) >= 0.5
        # accuracy
        post_cnn_correct = torch.eq(post_cnn_flat_bin_output, flat_labels.bool()).float()
        post_cnn_accuracy = torch.mean(post_cnn_correct)
        # precision, recall
        post_cnn_num_fg_output = torch.sum(post_cnn_flat_bin_output.float())
        post_cnn_tp = torch.sum(torch.logical_and(flat_labels.bool(), post_cnn_flat_bin_output).float())
        post_cnn_precision = torch.divide(post_cnn_tp,torch.add(post_cnn_num_fg_output, cfg.EPSILON))
        post_cnn_recall = torch.divide(post_cnn_tp,num_pixel_fg.float())


        return {'gnn_prob': gnn_prob, 'loss': loss, 'img_fg_prob': img_fg_prob,
                'cnn_loss': cnn_loss, 'cnn_accuracy': cnn_accuracy, 'cnn_precision': cnn_precision, 'cnn_recall': cnn_recall,
                'gnn_loss': gnn_loss, 'gnn_accuracy': gnn_accuracy,
                'post_cnn_img_fg_prob': post_cnn_img_fg_prob,
                'post_cnn_loss': post_cnn_loss, 'post_cnn_accuracy': post_cnn_accuracy,
                'post_cnn_precision': post_cnn_precision, 'post_cnn_recall': post_cnn_recall,
                'node_logits': node_logits, 'node_labels': node_labels}


class GAT(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def build_model(self):
        self.sp_attn_head_layer_1 = SPAttnHead(64, self.params.gat_hid_units[0],
                                               act=nn.ELU(),
                                               feat_dropout=self.params.gnn_feat_dropout,
                                               att_dropout=self.params.gnn_att_dropout,
                                               residual=self.params.gat_use_residual)

        self.sp_attn_head_layer_out = SPAttnHead(self.params.gat_hid_units[0] * self.params.gat_hid_units[0], 1,
                                                feat_dropout=self.params.gnn_feat_dropout,
                                                att_dropout=self.params.gnn_att_dropout,
                                                residual=False)

    def forward(self, node_byxs, adj, conv_feats, labels):
        node_feats = gather_nd(conv_feats, node_byxs)
        node_labels = torch.reshape(gather_nd(labels, node_byxs), [-1]).float()
        node_feats_resh = node_feats.unsqueeze(0)

        attns = []
        for _ in range(self.params.gat_n_heads[0]):
            attns.append(self.sp_attn_head_layer_1(node_feats_resh, adj))
        h_1 = torch.cat(attns, axis=1)

        out = self.sp_attn_head_layer_out(h_1, adj)
        for _ in range(self.params.gat_n_heads[-1] - 1):
            out =  out + self.sp_attn_head_layer_out(h_1, adj)

        node_logits = out / self.params.gat_n_heads[-1]
        node_logits = torch.squeeze(node_logits)

        ### Hang up the results ###
        gnn_final_feats = torch.squeeze(h_1)
        return (node_logits, gnn_final_feats, node_byxs, node_labels)



class SPAttnHead(nn.Module):
    def __init__(self, inchannel, out_channel,
                 act=None, feat_dropout=0., att_dropout=0., residual=False):
        super().__init__()
        self.out_channel = out_channel
        if feat_dropout != 0.0:
            self.feat_dropout = nn.Dropout(p=feat_dropout)
        if att_dropout != 0.0:
            self.att_dropout = nn.Dropout(p=att_dropout)

        self.fts_layer = nn.Conv1d(inchannel, out_channel, 1, bias=False)
        self.f1_layer = nn.Conv1d(out_channel, 1, 1)
        self.f2_layer = nn.Conv1d(out_channel, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2)
        if residual:
            self.residual_layer = nn.Conv1d(inchannel, 1, 1)
        self.vars_bias = nn.Parameter([], requires_grad=True)
        self.act = act

    def forward(self, x, adj):
        if hasattr(self, 'feat_dropout'):
            x = self.feat_dropout(x)
        fts = self.fts_layer(x)
        num_nodes = adj.size()[:1]
        f1 = self.f1_layer(fts)
        f2 = self.f2_layer(fts)
        # make shape match
        f1 = adj * f1
        f2 = adj * f2.permute([1, 0])
        # logits = f1.to_sparse() + f2.to_sparse()
        # lrelu = torch.sparse_coo_tensor(indices=logits.coalesce().indices(),
        #                                 values=self.lrelu(logits.coalesce().values()),
        #                                 size=logits.size())
        lrelu = self.lrelu(f1 + f2)

        coefs = torch.softmax(lrelu)
        if hasattr(self, 'att_dropout'):
            # coefs = torch.sparse_coo_tensor(indices=coefs.coalesce().indices(),
            #                                 values=self.att_dropout(coefs.coalesce().values()),
            #                                 size=coefs.size())
            coefs = self.att_dropout(coefs)

        if hasattr(self, 'feat_dropout'):
            fts = self.feat_dropout(fts)

        # coefs = tf.sparse_reshape(coefs, torch.cat([num_nodes,num_nodes], dim=0))
        coefs = torch.reshape(coefs, torch.cat([num_nodes,num_nodes], dim=0))
        fts = torch.squeeze(fts)
        # vals = tf.sparse_tensor_dense_matmul(coefs, fts)
        vals = torch.matmul(coefs, fts)
        vals = torch.reshape(vals, [1, num_nodes[:], self.out_channel])
        ret = vals + self.vars_bias

        if hasattr(self, 'residual_layer'):
            if x.size()[1] != ret.size()[1]:
                ret = ret + self.residual_layer(x)
            else:
                ret = ret + x

        return ret if self.act is None else self.act(ret)


def gather_nd(params, indices):
    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1) # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i] * m
        m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)



class Infer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        temp_num_chs = self.params.gat_n_heads[-2]*self.params.gat_hid_units[-1]
        self.post_cnn_conv_layer = new_conv_layer(temp_num_chs, 32, 1, 1, use_relu=True)
        self.upsampled_layer = new_deconv_layer(32, 16, 4, stride=2)
        if params.use_enc_layer:
            self.cur_cnn_layer = new_conv_layer(16, 16, 1, 1, use_relu=True)

        self.output_layer_in = new_conv_layer(32, 32, self.params.infer_module_kernel_size, 1, use_relu=True)
        self.output_layer_out = new_conv_layer(32, 1, self.params.infer_module_kernel_size, 1, use_relu=True)


    def forward(self, imgs, cnn_feat, gnn_final_feats,
                is_lr_flipped, is_ud_flipped):
        y_len = imgs.size()[-2] // self.params.win_size
        x_len = imgs.size()[-1] // self.params.win_size

        # (9216, 64) -> [ 1 96 96 64]
        reshaped_gnn_feats = torch.reshape(gnn_final_feats, (-1, gnn_final_feats.size()[1], y_len, x_len))

        if is_lr_flipped:
            reshaped_gnn_feats = reshaped_gnn_feats[:, :, :, ::-1]
        if is_ud_flipped:
            reshaped_gnn_feats = reshaped_gnn_feats[:, :, ::-1, :]

        post_cnn_conv_comp = self.post_cnn_conv_layer(reshaped_gnn_feats)
        current_input = post_cnn_conv_comp
        ds_rate = self.params.win_size // 2
        while ds_rate >= 1:
            upsampled = self.upsampled_layer(current_input)
            cur_cnn_feat = torch.dropout(cnn_feat[ds_rate], 1 - self.params.post_cnn_dropout)
            if hasattr(self, 'cur_cnn_layer'):
                cur_cnn_feat = self.cur_cnn_layer(cur_cnn_feat)
            if ds_rate == 1:
                output = self.output_layer_out(torch.cat([upsampled,cur_cnn_feat], dim=1))
                post_cnn_img_output = output
            else:
                output = self.output_layer_in(torch.cat([upsampled,cur_cnn_feat], dim=1))
            current_input = output
            ds_rate = ds_rate // 2

        post_cnn_img_fg_prob = torch.sigmoid(current_input)

        return (post_cnn_img_fg_prob, post_cnn_img_output)