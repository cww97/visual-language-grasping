from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
from torch.autograd import Variable
import torch


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


class Rua(nn.Module):
	pass