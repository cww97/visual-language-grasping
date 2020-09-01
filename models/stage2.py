from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
from torch.autograd import Variable
import torch
from torchvision.models import resnet50
from collections import OrderedDict
from models.attentions import DotAttention


class EncoderLSTM(nn.Module):
	''' https://github.com/peteanderson80/Matterport3DSimulator
		Encodes navigation instructions, returning hidden state context (for
		attention methods) and a decoder initial state. '''

	def __init__(self, vocab_size, padding_idx, embedding_size=128, hidden_size=256,
					dropout_ratio=0.1, bidirectional=False, num_layers=1):
		super(EncoderLSTM, self).__init__()
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


class RgbdCNN(nn.Module):

	def __init__(self, output_dim=1024):
		super().__init__()
		self.output_dim = output_dim

		self.color_encoder = self._resnet_feature_extractor()
		self.depth_encoder = self._resnet_feature_extractor()
		self.bus = nn.Sequential(OrderedDict([
			('conv1', nn.Conv2d(4096, 2048, kernel_size=1, stride=1, bias=False)),
			('bn1', nn.BatchNorm2d(2048)),
			('conv2', nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=False)),
			('bn2', nn.BatchNorm2d(2048)),
			('conv3', nn.Conv2d(2048, 4096, kernel_size=1, stride=1, bias=False)),
			('bn3', nn.BatchNorm2d(4096)),
			('relu', nn.ReLU(inplace=True)),
			('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
		]))
		self.fc = nn.Linear(4096, output_dim)

	def _resnet_feature_extractor(self):
		resnet = resnet50(pretrained=True)
		modules = list(resnet.children())[:-2]
		resnet = nn.Sequential(*modules)
		for param in resnet.parameters():
			param.requires_grad = False
		return resnet

	def forward(self, candidates):
		# get features of imgs
		batch_size, max_k, c, h, w = candidates.shape
		candidates = candidates.view(-1, c, h, w)
		color_imgs = candidates[:, :3, :, :]
		depth_imgs = candidates[:, 3:, :, :]
		# depth_imgs = candidates[:, -1:, :, :].repeat(1, 3, 1, 1)
		
		color_feats = self.color_encoder(color_imgs)
		depth_feats = self.depth_encoder(depth_imgs)
		x = torch.cat((color_feats, depth_feats), dim=1)
		x = self.bus(x)
		x = x.reshape(x.size(0), -1)
		x = self.fc(x)
		
		return x.reshape(batch_size, max_k, self.output_dim)


class SelectNet(nn.Module):

	def __init__(self, **kwargs):
		super().__init__()
		self.text_encoder = EncoderLSTM(**kwargs, hidden_size=256)
		self.img0_encoder = RgbdCNN(output_dim=1024)
		# self.img1_encoder = RgbdCNN(output_dim=1024)
		self.attention = DotAttention(text_dim=256, imgs_dim=1024)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, instructions, candidates):
		'''
		instructions:
		candidates: [batch_size, k, c, h, w]
		'''
		instr_tensors = torch.cat(instructions.tensor).cuda()
		instr_lengths = torch.Tensor(instructions.length).cuda()
		text_feats, _, _ = self.text_encoder(instr_tensors, instr_lengths)  # [b, len, dim_1]

		imgs_feats = self.img0_encoder(candidates[0])
		# img1_feats = self.img1_encoder(candidates[1])
		is_valid = candidates[1]
		# imgs_feats = torch.cat((img0_feats, img1_feats))  # ?

		scores = self.attention(text_feats, imgs_feats)
		# import pdb; pdb.set_trace()
		scores[is_valid == 0] = -float('inf')
		pred = self.softmax(scores)

		return pred
