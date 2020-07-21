# !/usr/bin/env python

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import namedtuple
from envs.data import Instruction
State = namedtuple('State', ('instruction', 'color_data', 'depth_data', 'widths'))


class EncoderLSTM(nn.Module):
	''' https://github.com/peteanderson80/Matterport3DSimulator
		Encodes navigation instructions, returning hidden state context (for
		attention methods) and a decoder initial state. '''

	def __init__(self, vocab_size, padding_idx, embedding_size=256, hidden_size=512,
				 dropout_ratio=0.1, bidirectional=False, num_layers=1):
		super(EncoderLSTM, self).__init__()
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.drop = nn.Dropout(p=dropout_ratio)
		self.num_directions = 2 if bidirectional else 1
		self.num_layers = num_layers
		self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
		self.lstm = nn.LSTM(embedding_size, hidden_size, self.num_layers,
							batch_first=True,  # dropout=dropout_ratio,
							bidirectional=bidirectional)
		self.encoder2decoder = nn.Linear(
			hidden_size * self.num_directions,
			hidden_size * self.num_directions
		)

	def init_state(self, inputs):
		''' Initialize to zero cell states and hidden states.'''
		batch_size = inputs.size(0)
		h0 = Variable(torch.zeros(
			self.num_layers * self.num_directions,
			batch_size, self.hidden_size
		), requires_grad=False)
		c0 = Variable(torch.zeros(
			self.num_layers * self.num_directions,
			batch_size, self.hidden_size
		), requires_grad=False)
		return h0.cuda(), c0.cuda()

	def forward(self, inputs, lengths):
		''' Expects input vocab indices as (batch, seq_len). Also requires a
			list of lengths for dynamic batching. '''
		if self.embedding.weight.device.type == 'cpu':
			import pdb; pdb.set_trace()
		embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
		embeds = self.drop(embeds)
		packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
		h0, c0 = self.init_state(inputs)
		enc_h, (enc_h_t, enc_c_t) = self.lstm(packed_embeds, (h0, c0))

		if self.num_directions == 2:
			h_t = torch.cat((enc_h_t[-1], enc_h_t[-2]), 1)
			c_t = torch.cat((enc_c_t[-1], enc_c_t[-2]), 1)
		else:
			h_t = enc_h_t[-1]
			c_t = enc_c_t[-1]  # (batch, hidden_size)

		decoder_init = nn.Tanh()(self.encoder2decoder(h_t))

		ctx, lengths = pad_packed_sequence(enc_h, batch_first=True)
		# ctx = self.drop(ctx)
		return ctx, decoder_init, c_t  # (batch, seq_len, hidden_size*num_directions)
		# (batch, hidden_size)


class BiAttention(nn.Module):
	'''Bilinear Pooling Attention
	'''
	def __init__(self, img_dim, word_dim, hidden_dim=1024):
		super().__init__()
		self.fc_1_a = nn.Linear(img_dim, hidden_dim, bias=False)
		self.fc_1_b = nn.Linear(word_dim, hidden_dim, bias=False)
		self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc_3 = nn.Linear(hidden_dim, 1)

	def forward(self, img_feat_map, word_feat_vec):
		'''
		feature_map:     batch * c * w * h
		feature_vector:  batch * d
		'''
		# print(img_feat_map.shape, word_feat_vec.shape)
		batch_size = img_feat_map.size()[0]
		conv_size = img_feat_map.size()[3]  # w == h
		
		# wh_feat = torch.transpose(torch.transpose(img_feat_map, 1, 2), 2, 3)  
		wh_feat = img_feat_map.permute(0, 2, 3, 1)  # reshape (b, w, h, c)
		wh_feat = wh_feat.contiguous().view(batch_size*conv_size*conv_size, -1)  # [b*w*h, c]
		wh_feat = self.fc_1_a(wh_feat)  # [b*w*h, hidden_dim]: [400, 1024]
		wd_feat = self.fc_1_b(word_feat_vec).repeat_interleave(conv_size*conv_size, 0)  # [b*w*h, hidden_dim:1024]
		kx_feat = self.fc_2(torch.tanh(wh_feat * wd_feat))  # zkx feature, [400, 1024]

		atten = self.fc_3(kx_feat)  # [400, 1], now attention
		atten = atten.view(batch_size, conv_size, conv_size, 1).permute(0, 3, 1, 2)  # reshape back
		img_feat_map *= atten.repeat([1, 2048, 1, 1])  # to mul
		# print(img_feat_map.size(), atten.size()); assert False
		return img_feat_map, atten


class reinforcement_net(nn.Module):

	def __init__(self, **kwargs):
		super().__init__()
		self.text_encoder = EncoderLSTM(**kwargs)

		# Initialize network trunks with DenseNet pre-trained on ImageNet
		self.grasp_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
		self.grasp_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)

		self.num_rotations = 16

		self.attention_layer = BiAttention(img_dim=2048, word_dim=self.text_encoder.hidden_size)
		# Construct network branches for xxshing and grasping
		self.graspnet = nn.Sequential(OrderedDict([
			('grasp-norm0', nn.BatchNorm2d(2048)),
			('grasp-relu0', nn.ReLU(inplace=True)),
			('grasp-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
			('grasp-norm1', nn.BatchNorm2d(64)),
			('grasp-relu1', nn.ReLU(inplace=True)),
			('grasp-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
			# ('grasp-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
		]))
		self.unsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

		# Initialize network weights
		for m in self.named_modules():
			if 'push-' in m[0] or 'grasp-' in m[0]:
				if isinstance(m[1], nn.Conv2d):
					nn.init.kaiming_normal_(m[1].weight.data)
				elif isinstance(m[1], nn.BatchNorm2d):
					m[1].weight.data.fill_(1)
					m[1].bias.data.zero_()

		# Compute sample grid for rotation BEFORE neural network
		prev_rotate_mats = []
		post_rotate_mats = []
		for rotate_idx in range(self.num_rotations):
			theta = np.radians(rotate_idx * (360 / self.num_rotations))
			prev_rotate_mats.append(self._compute_rotate_mat(-theta, [1, 3, 640, 640]))
			post_rotate_mats.append(self._compute_rotate_mat(theta, [1, 2048, 20, 20]))
		self.prev_rotate_mats = torch.cat(prev_rotate_mats)
		self.post_rotate_mats = torch.cat(post_rotate_mats)
		# import pdb; pdb.set_trace()encoding

	def _compute_rotate_mat(self, rotate_theta, size):
		affine_mat = np.asarray([
			[np.cos(rotate_theta), np.sin(rotate_theta), 0],
			[-np.sin(rotate_theta), np.cos(rotate_theta), 0]
		])
		affine_mat.shape = (2, 3, 1)
		affine_mat = torch.from_numpy(affine_mat).permute(2, 0, 1).float()
		affine_mat = Variable(affine_mat, requires_grad=False).cuda()
		flow_grid = F.affine_grid(affine_mat, size)
		return flow_grid

	def train(self, *args):
		super().train(*args)
		self.grasp_color_trunk.eval()
		self.grasp_depth_trunk.eval()

	def forward(self, states, rotations=[]):
		'''
		3 stituations:
			- sample: batch_size == 1, 16 rotations(len(rotations) == 0)
			- target_net: batch forward, 16 rotations(len(rotations) == 0)
			- policy_net: batch forward, 1 roatation
		'''
		# unpack
		batch_size = len(states)
		states = State(*zip(*states))
		if len(rotations) == 0:  # for 16 rotations
			num_rotations = 16
			prev_rotate_mats = self.prev_rotate_mats.repeat(batch_size, 1, 1, 1)
			post_rotate_mats = self.post_rotate_mats.repeat(batch_size, 1, 1, 1)
		else:
			num_rotations = 1
			prev_rotate_mats = self.prev_rotate_mats.index_select(0, rotations)
			post_rotate_mats = self.post_rotate_mats.index_select(0, rotations)
		
		instructions = Instruction(*zip(*states.instruction))
		instr_tensors = torch.cat([i.repeat(num_rotations, 1) for i in instructions.tensor]).cuda()  # [B*16, d]
		instr_lengths = torch.tensor([[l]*num_rotations for l in instructions.length]).flatten().cuda()
		color_datas = torch.cat([c.repeat(num_rotations, 1, 1, 1) for c in states.color_data]).cuda()
		depth_datas = torch.cat([d.repeat(num_rotations, 1, 1, 1) for d in states.depth_data]).cuda()
		widths = torch.tensor(states.widths[0]).cuda()

		text_feats, _, _ = self.text_encoder(instr_tensors, instr_lengths)  # (N, d)
		text_feats = text_feats[:, -1, :]
		rotate_colors = F.grid_sample(color_datas, prev_rotate_mats, mode='nearest')
		rotate_depths = F.grid_sample(depth_datas, prev_rotate_mats, mode='nearest')
		with torch.no_grad():
			color_feats = self.grasp_color_trunk.features(rotate_colors)
			depth_feats = self.grasp_depth_trunk.features(rotate_depths)
			image_feats = torch.cat((color_feats, depth_feats), dim=1)
		atten_feats, _ = self.attention_layer(image_feats, text_feats)
		grasp_preds = self.graspnet(atten_feats)
		grasp_preds = F.grid_sample(grasp_preds, post_rotate_mats, mode='nearest')
		grasp_preds = self.unsample(grasp_preds)
		grasp_preds = grasp_preds[:, 0, widths[0]:widths[1], widths[0]:widths[1]]
		return grasp_preds


class CNN_Text(nn.Module):
    '''https://github.com/Shawn1993/cnn-text-classification-pytorch
    '''

    def __init__(self, vocab_size, padding_idx, embed_dim=256, drop_out=0.5):
        super().__init__()
        # print(embed_num, embed_dim, drop_out)
        self.embed_num = vocab_size
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(self.embed_num, self.embed_dim)

        Ci = 1  # total 1024 = 32 * 32
        self.conv1_1 = nn.Conv2d(Ci, 81920, (1, self.embed_dim))
        self.conv1_2 = nn.Conv2d(Ci, 81920, (2, self.embed_dim))
        self.conv1_3 = nn.Conv2d(Ci, 81920, (3, self.embed_dim))
        self.conv1_4 = nn.Conv2d(Ci, 81920, (4, self.embed_dim))
        self.conv1_5 = nn.Conv2d(Ci, 81920, (5, self.embed_dim))
        self.dropout = nn.Dropout(drop_out)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)              # (N, W, D)
        x = Variable(x).unsqueeze(1)   # (N, Ci, W, D)

        x1 = self.conv_and_pool(x, self.conv1_1)
        x2 = self.conv_and_pool(x, self.conv1_2)
        x3 = self.conv_and_pool(x, self.conv1_3)
        x4 = self.conv_and_pool(x, self.conv1_4)
        x5 = self.conv_and_pool(x, self.conv1_5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)  # (N,1024)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = x.reshape(len(x), 1024, 20, 20)
        # print('end ', x.shape, type(x)); assert False
        return x


if __name__ == '__main__':
    example_instruction = 'pick up the {color} {shape}.'
