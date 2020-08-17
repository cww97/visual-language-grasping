# !/usr/bin/env python

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import namedtuple
from envs.data import Instruction
State = namedtuple('State', ('instruction', 'color_data', 'depth_data'))
from models.stage1 import Seg
from models.stage2 import EncoderLSTM

class YoungNet(nn.Module):

	def __init__(self, **kwargs):
		super().__init__()
		self.text_encoder = EncoderLSTM(**kwargs)

		self.num_rotations = 16
		self.obj_finder = Seg()

	def forward(self, states):
		batch_size = len(states)
		states = State(*zip(*states))
		instructions = Instruction(*zip(*states.instruction))

		# in stage2
		instr_tensors = torch.cat(instructions.tensor).cuda()
		instr_lengths = torch.Tensor(instructions.length).cuda()

		candidates, is_valid = self.obj_finder(states.color_data, states.depth_data)
		

		import pdb; pdb.set_trace()
