# !/usr/bin/env python

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CNN_Text(nn.Module):
    '''https://github.com/Shawn1993/cnn-text-classification-pytorch
    '''

    def __init__(self, embed_num, embed_dim, drop_out):
        super().__init__()
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(self.embed_num, self.embed_dim)

        Ci = 1  # total 1024 = 32 * 32
        self.conv1_1 = nn.Conv2d(Ci, 104, (1, self.embed_dim))
        self.conv1_2 = nn.Conv2d(Ci, 104, (2, self.embed_dim))
        self.conv1_3 = nn.Conv2d(Ci, 104, (3, self.embed_dim))
        self.conv1_4 = nn.Conv2d(Ci, 104, (4, self.embed_dim))
        self.conv1_5 = nn.Conv2d(Ci, 104, (5, self.embed_dim))
        self.conv1_6 = nn.Conv2d(Ci, 104, (6, self.embed_dim))
        self.conv1_7 = nn.Conv2d(Ci, 100, (7, self.embed_dim))
        self.conv1_8 = nn.Conv2d(Ci, 100, (8, self.embed_dim))
        self.conv1_9 = nn.Conv2d(Ci, 100, (9, self.embed_dim))
        self.conv1_10 = nn.Conv2d(Ci, 100, (10, self.embed_dim))

        self.dropout = nn.Dropout(drop_out)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # x = x[:1, ]
        x = self.embed(x)              # (N, W, D)
        print('begin: ', x.shape)
        x = Variable(x).unsqueeze(1)   # (N, Ci, W, D)

        x1 = self.conv_and_pool(x, self.conv1_1)
        x2 = self.conv_and_pool(x, self.conv1_2)
        x3 = self.conv_and_pool(x, self.conv1_3)
        x4 = self.conv_and_pool(x, self.conv1_4)
        x5 = self.conv_and_pool(x, self.conv1_5)
        x6 = self.conv_and_pool(x, self.conv1_6)
        x7 = self.conv_and_pool(x, self.conv1_7)
        x8 = self.conv_and_pool(x, self.conv1_8)
        x9 = self.conv_and_pool(x, self.conv1_9)
        x10 = self.conv_and_pool(x, self.conv1_10)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10), 1)  # (N,1024)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = x.reshape(len(x), 32, 32)
        # print('end ', x.shape, type(x)); assert False
        return x


class BaseNet(nn.Module):

    def __init__(self, grasp_conv1_out_channels=3, **kwargs):
        super().__init__()
        # self.text_encoder = CNN_Text(**kwargs)

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.grasp_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.grasp_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)

        self.num_rotations = 16

        # Construct network branches for xxshing and grasping
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(2048)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, grasp_conv1_out_channels, kernel_size=1, stride=1, bias=False))
            # ('grasp-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0] or 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []

    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):
        """
        is_volatile: true for each rotation, false for specific_rotation
        use is_volatile=false in backward
        """
        # TODO: rua
        # text_feature = self.text_encoder(instructor)

        if is_volatile:  # try every rotate angle
            rotations = range(self.num_rotations)
            torch.set_grad_enabled(False)
            output_prob = []
            interm_feat = []
        else:
            rotations = [specific_rotation]
            output_prob = self.output_prob = []
            interm_feat = self.interm_feat = []

        # Apply rotations to images
        for rotate_idx in rotations:
            rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE neural network
            affine_mat_before = np.asarray([
                [np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]
            ])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            flow_grid_before = F.affine_grid(
                Variable(affine_mat_before, requires_grad=False).cuda(),
                input_color_data.size()
            )

            # Rotate images clockwise
            rotate_color = F.grid_sample(
                Variable(input_color_data, requires_grad=False).cuda(),
                flow_grid_before, mode='nearest'
            )
            rotate_depth = F.grid_sample(
                Variable(input_depth_data, requires_grad=False).cuda(),
                flow_grid_before, mode='nearest'
            )

            # Compute intermediate features, use pretrained densenet
            interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
            interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
            interm_grasp_feat = torch.cat(
                (interm_grasp_color_feat, interm_grasp_depth_feat), dim=1
            )
            interm_feat.append([interm_grasp_feat])

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([
                [np.cos(rotate_theta), np.sin(rotate_theta), 0],
                [-np.sin(rotate_theta), np.cos(rotate_theta), 0]
            ])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()

            flow_grid_after = F.affine_grid(
                Variable(affine_mat_after, requires_grad=False).cuda(),
                interm_grasp_feat.data.size()
            )

            # TODO : rua
            # Forward pass through branches, undo rotation on output predictions, upsample results
            output_prob.append([
                nn.Upsample(
                    scale_factor=16, mode='bilinear', align_corners=True
                ).forward(F.grid_sample(
                    self.graspnet(interm_grasp_feat),
                    flow_grid_after, mode='nearest'
                ))
            ])

        if is_volatile: torch.set_grad_enabled(True)
        return output_prob, interm_feat


class reactive_net(BaseNet):
    def __init__(self, **kwargs):
        super().__init__(grasp_conv1_out_channels=3)


class reinforcement_net(BaseNet):
    def __init__(self):
        super().__init__(grasp_conv1_out_channels=1)


if __name__ == '__main__':
    example_instruction = 'pick up the {color} {shape}.'
