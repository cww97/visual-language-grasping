# !/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import namedtuple
from envs.data import Instruction
State = namedtuple('State', ('instruction', 'color_data', 'depth_data'))

from models.stage1 import Seg
from models.stage2 import SelectNet

class YoungNet(nn.Module):

	def __init__(self, **kwargs):
		super().__init__()
		self.num_rotations = 16
		self.obj_finder = Seg()
		self.obj_selector = SelectNet(**kwargs)

	def forward(self, states):
		batch_size = len(states)
		states = State(*zip(*states))
		instructions = Instruction(*zip(*states.instruction))

		candidate_objs = self.obj_finder(states.color_data, states.depth_data)
		choosen_obj = self.obj_selector(instructions, candidate_objs)

		import pdb; pdb.set_trace()

		return 0
