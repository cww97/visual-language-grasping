import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.autograd import Variable

from envs.data import Data as TextData
from envs.robot import Reward
from models import reinforcement_net
from utils import CrossEntropyLoss2d
from collections import namedtuple
import random
import math

State = namedtuple('State', ('instruction', 'color_data', 'depth_data', 'widths'))
Action = namedtuple('Action', ('r', 'x', 'y'))
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Trainer(object):

	def __init__(self, future_reward_discount, load_snapshot, snapshot_file):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		self.text_data = TextData()
		vocab, pad_idx = len(self.text_data.text_field.vocab), self.text_data.padding_idx

		self.model = reinforcement_net(vocab_size=vocab, padding_idx=pad_idx).to(self.device)
		if load_snapshot:  # Load pre-trained model
			self.model.load_state_dict(torch.load(snapshot_file))
			print('Pre-trained model snapshot loaded from: %s' % (snapshot_file))
		self.model.train()  # Set model to training mode
		self.target_net = reinforcement_net(vocab_size=vocab, padding_idx=pad_idx).to(self.device)
		self.target_net.load_state_dict(self.model.state_dict())
		self.target_net.eval()

		self.future_reward_discount = future_reward_discount
		self.criterion = torch.nn.SmoothL1Loss(reduction='none').cuda()  # Initialize Huber loss

		# Initialize optimizer
		self.optimizer = torch.optim.Adam(self.model.parameters())
		# self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
		self.BATCH_SIZE = 2
		self.memory = ReplayMemory(256)
		self.iteration = 0

		# Initialize lists to save execution info and RL variables
		self.action_log = []
		self.reward_log = []

	# Pre-load execution info and RL variables
	def preload(self, transitions_directory):
		self.iteration = self.action_log.shape[0] - 2
		self.action_log = np.loadtxt(os.path.join(transitions_directory, 'action.log.txt'), delimiter=' ')
		self.action_log = self.action_log[0:self.iteration, :]
		self.action_log = self.action_log.tolist()
		self.reward_log = np.loadtxt(os.path.join(transitions_directory, 'reward.log.txt'), delimiter=' ')
		self.reward_log.shape = (self.iteration, 1)
		self.reward_log = self.reward_log.tolist()

	def _get_eps_threshold(self, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200):
		return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.iteration / EPS_DECAY)

	def select_action(self, state, env=None):
		sample = random.random()
		eps_threshold = self._get_eps_threshold()
		self.iteration += 1

		if sample > eps_threshold:
			with torch.no_grad():
				grasp_pred = self.model([state]).cpu().data.numpy()
				action = np.unravel_index(np.argmax(grasp_pred), grasp_pred.shape)
				choice = 'policy_network'
		else:
			grasp_pred = None
			action = env.random_grasp_action()
			choice = 'random_select'

		return choice, action, grasp_pred

	def optimize_model(self):
		if len(self.memory) < self.BATCH_SIZE: return None
		transitions = self.memory.sample(self.BATCH_SIZE)
		# batch = Transition(*zip(*transitions))
		# import pdb; pdb.set_trace()
		loss = 0.0
		for transition in transitions:
			loss += self.backprop(*transition)
		loss /= self.BATCH_SIZE
		print("model updated, loss = %.4f" % (loss))
		return loss

	# Compute labels and back-propagate
	def backprop(self, state, action, next_states, reward):
		with torch.no_grad():
			next_grasp_pred = self.target_net([next_states])
			future_reward = torch.max(next_grasp_pred)
			expected_reward = reward + self.future_reward_discount * future_reward

		# Compute loss and backward pass
		self.optimizer.zero_grad()
		loss_value = 0
		# Do for-ward pass with specified rotation (to save gradients)
		# rotations = [action[0], (action[0] + self.model.num_rotations // 2) % self.model.num_rotations]
		loss_value += self._backward_loss([state], [action[0]], expected_reward)
		opposite_rotate_idx = (action[0] + self.model.num_rotations // 2) % self.model.num_rotations
		
		# import pdb; pdb.set_trace()
		loss_value += self._backward_loss([state], [opposite_rotate_idx], expected_reward)
		loss_value /= 2

		self.optimizer.step()
		return loss_value

	def _backward_loss(self, state_batch, rotations, expected_state_action_values):
		# import pdb; pdb.set_trace()
		
		state_action_values = torch.max(self.model(state_batch, rotations))
		loss = self.criterion(state_action_values, expected_state_action_values)
		loss = loss.sum()
		loss.backward()
		return loss.cpu().data.numpy()

	def get_pred_vis(self, predictions, color_map, best_pix):
		canvas = None
		num_rotations = predictions.shape[0]
		for canvas_row in range(int(num_rotations / 4)):
			tmp_row_canvas = None
			for canvas_col in range(4):
				rotate_idx = canvas_row * 4 + canvas_col
				pred_vis = predictions[rotate_idx, :, :].copy()
				pred_vis.shape = (predictions.shape[1], predictions.shape[2])
				pred_vis = (np.clip(pred_vis, 0, 1) * 255).astype(np.uint8)
				pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)
				if rotate_idx == best_pix[0]:
					pix_idx = (int(best_pix[2]), int(best_pix[1]))
					pred_vis = cv2.circle(pred_vis, pix_idx,7, (0, 0, 255), 2)
				r_angle = rotate_idx * (360.0 / num_rotations)
				pred_vis = ndimage.rotate(pred_vis, r_angle, reshape=False, order=0)
				
				background_image = ndimage.rotate(color_map, r_angle, reshape=False, order=0)
				background_image = cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR)

				pred_vis = (0.5 * background_image + 0.5 * pred_vis).astype(np.uint8)

				if tmp_row_canvas is None: tmp_row_canvas = pred_vis
				else: tmp_row_canvas = np.concatenate((tmp_row_canvas, pred_vis), axis=1)
			if canvas is None: canvas = tmp_row_canvas
			else: canvas = np.concatenate((canvas, tmp_row_canvas), axis=0)
		return canvas


if __name__ == '__main__':
	loaded_snapshot_state_dict = torch.load('downloads/vpg-original-real-pretrained-30-obj.pth')
	with open('model_dict.txt', 'w') as f:
		f.write(str(loaded_snapshot_state_dict.keys()))
