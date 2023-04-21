import torch.nn as nn
import torch


GN_MIN_NUM_G = 8
GN_MIN_CHS_PER_G = 16





class DRIU(nn.Module):
    def __init__(self, for_vgn=False):
        super().__init__()
        self.for_vgn = for_vgn
        mid_channel = 16
        self.conv1_1 = new_conv_layer(3, 64, use_relu=True)
        self.conv1_2 = new_conv_layer(64, 64, use_relu=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = new_conv_layer(64, 128, use_relu=True)
        self.conv2_2 = new_conv_layer(128, 128, use_relu=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = new_conv_layer(128, 256, use_relu=True)
        self.conv3_2 = new_conv_layer(256, 256, use_relu=True)
        self.conv3_3 = new_conv_layer(256, 256, use_relu=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_1 = new_conv_layer(256, 512, use_relu=True)
        self.conv4_2 = new_conv_layer(512, 512, use_relu=True)
        self.conv4_3 = new_conv_layer(512, 512, use_relu=True)

        self.spe_1 = new_conv_layer(64, mid_channel, use_relu=True)
        self.spe_2 = new_conv_layer(128, mid_channel, use_relu=True)
        self.up_2 = new_deconv_layer(mid_channel, mid_channel, ksize=4, stride=2, padding=1, use_relu=True)
        self.spe_3 = new_conv_layer(256, mid_channel, use_relu=True)
        self.up_3 = new_deconv_layer(mid_channel, mid_channel, ksize=8, stride=4, padding=2, use_relu=True)
        self.spe_4 = new_conv_layer(512, mid_channel, use_relu=True)
        self.up_4 = new_deconv_layer(mid_channel, mid_channel, ksize=16, stride=8, padding=4, use_relu=True)
        self.output = new_conv_layer(mid_channel*4, 1, 1, padding=0)

    def forward(self, x):
        conv1_2 = self.conv1_2(self.conv1_1(x))
        pool1 = self.pool1(conv1_2)
        conv2_2 = self.conv2_2(self.conv2_1(pool1))
        pool2 = self.pool2(conv2_2)
        conv3_3 = self.conv3_3(self.conv3_2(self.conv3_1(pool2)))
        pool3 = self.pool3(conv3_3)
        conv4_3 = self.conv4_3(self.conv4_2(self.conv4_1(pool3)))

        spe1 = self.spe_1(conv1_2)
        spe2 = self.spe_2(conv2_2)
        resized_spe_2 = self.up_2(spe2)
        spe3 = self.spe_3(conv3_3)
        resized_spe_3 = self.up_3(spe3)
        spe4 = self.spe_4(conv4_3)
        resized_spe_4 = self.up_4(spe4)

        conv_feats = torch.cat([spe1, resized_spe_2, resized_spe_3, resized_spe_4], dim=1)
        img_output = self.output(conv_feats)
        if not self.for_vgn: return img_output
        cnn_feat = {
            1: spe1,
            2: spe2,
            4: spe3,
            8: spe4
        }
        return (cnn_feat, conv_feats, img_output)


class LargeDRIU(nn.Module):
    def __init__(self, for_vgn=False):
        super().__init__()
        self.for_vgn = for_vgn
        mid_channel = 16
        self.conv1_1 = new_conv_layer(3, 64, use_relu=True)
        self.conv1_2 = new_conv_layer(64, 64, use_relu=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = new_conv_layer(64, 128, use_relu=True)
        self.conv2_2 = new_conv_layer(128, 128, use_relu=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = new_conv_layer(128, 256, use_relu=True)
        self.conv3_2 = new_conv_layer(256, 256, use_relu=True)
        self.conv3_3 = new_conv_layer(256, 256, use_relu=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_1 = new_conv_layer(256, 512, use_relu=True)
        self.conv4_2 = new_conv_layer(512, 512, use_relu=True)
        self.conv4_3 = new_conv_layer(512, 512, use_relu=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = new_conv_layer(512, 512, use_relu=True)
        self.conv5_2 = new_conv_layer(512, 512, use_relu=True)
        self.conv5_3 = new_conv_layer(512, 512, use_relu=True)

        self.spe_1 = new_conv_layer(64, mid_channel, use_relu=True)
        self.spe_2 = new_conv_layer(128, mid_channel, use_relu=True)
        self.up_2 = new_deconv_layer(mid_channel, mid_channel, ksize=4, stride=2, padding=1, use_relu=True)
        self.spe_3 = new_conv_layer(256, mid_channel, use_relu=True)
        self.up_3 = new_deconv_layer(mid_channel, mid_channel, ksize=8, stride=4, padding=2, use_relu=True)
        self.spe_4 = new_conv_layer(512, mid_channel, use_relu=True)
        self.up_4 = new_deconv_layer(mid_channel, mid_channel, ksize=16, stride=8, padding=4, use_relu=True)
        self.spe_5 = new_conv_layer(512, mid_channel, use_relu=True)
        self.up_5 = new_deconv_layer(mid_channel, mid_channel, ksize=32, stride=16, padding=8, use_relu=True)
        self.output = new_conv_layer(mid_channel*5, 1, 1, padding=0)

    def forward(self, x):
        conv1_2 = self.conv1_2(self.conv1_1(x))
        pool1 = self.pool1(conv1_2)
        conv2_2 = self.conv2_2(self.conv2_1(pool1))
        pool2 = self.pool2(conv2_2)
        conv3_3 = self.conv3_3(self.conv3_2(self.conv3_1(pool2)))
        pool3 = self.pool3(conv3_3)
        conv4_3 = self.conv4_3(self.conv4_2(self.conv4_1(pool3)))
        pool4 = self.pool4(conv4_3)
        conv5_3 = self.conv5_3(self.conv5_2(self.conv5_1(pool4)))

        spe1 = self.spe_1(conv1_2)
        spe2 = self.spe_2(conv2_2)
        resized_spe_2 = self.up_2(spe2)
        spe3 = self.spe_3(conv3_3)
        resized_spe_3 = self.up_3(spe3)
        spe4 = self.spe_4(conv4_3)
        resized_spe_4 = self.up_4(spe4)
        spe5 = self.spe_5(conv5_3)
        resized_spe_5 = self.up_5(spe5)
        spe_concat = torch.cat([spe1, resized_spe_2, resized_spe_3, resized_spe_4, resized_spe_5], dim=1)
        output = self.output(spe_concat)
        if not self.for_vgn: return output
        cnn_feat = {
            1: spe1,
            2: spe2,
            4: spe3,
            8: spe4,
            16: spe5
        }
        return (cnn_feat, spe_concat, output)



def new_conv_layer(in_channel, out_channel, ksize=3, stride=1, padding=1, use_relu=False):
    norm = nn.BatchNorm2d(out_channel)
    conv = nn.Conv2d(in_channel,
                     out_channel,
                     kernel_size=ksize,
                     stride=stride,
                     padding=padding,
                     bias=True)
    layers = [conv, norm]
    layers.append(norm)
    if use_relu: layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def new_deconv_layer(in_channel, out_channel, ksize=3, stride=1, padding=1,
                     use_relu=False):
    norm = nn.BatchNorm2d(out_channel)
    dconv = nn.ConvTranspose2d(in_channel, out_channel,
                               kernel_size=ksize, stride=stride,
                               padding=padding, bias=True)
    layers = [dconv, norm]
    layers.append(norm)
    if use_relu: layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


