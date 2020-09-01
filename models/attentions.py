import torch
import torch.nn as nn


class DotAttention(nn.Module):
	def __init__(self, text_dim=256, imgs_dim=1024, hidden_dim=512):
		super().__init__()
		self.text_in = nn.Linear(text_dim, hidden_dim, bias=False)
		self.imgs_in = nn.Linear(imgs_dim, hidden_dim, bias=False)
		self.tanh = nn.Tanh()

	def forward(self, text_feats, imgs_feats):
		'''
		imgs_feats: [batch_size, k, dim_1], where k is the number of candidates
		text_feats: [batch_size, len, dim_2], LSTM outputs

		return:     [batch_size, k], scores for each candidates
		'''
		text_feats = text_feats[:, -1, :]
		text_feats = self.text_in(text_feats)
		
		batch_size, k, dim1 = imgs_feats.shape
		imgs_feats = imgs_feats.reshape(batch_size*k, -1)
		imgs_feats = self.imgs_in(imgs_feats)
		imgs_feats = imgs_feats.reshape(batch_size, k, -1)

		scores = torch.bmm(imgs_feats, text_feats.unsqueeze(2)).squeeze(2)
		scores = self.tanh(scores)
		return scores.clone()