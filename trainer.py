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

State = namedtuple('State', ('instruction', 'color_data', 'depth_data'))
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
		assert torch.cuda.is_available()
		self.text_data = TextData()
		vocab, pad_idx = len(self.text_data.text_field.vocab), self.text_data.padding_idx

		self.model = reinforcement_net(vocab_size=vocab, padding_idx=pad_idx)  # 'reinforcement'
		if load_snapshot:  # Load pre-trained model
			self.model.load_state_dict(torch.load(snapshot_file))
			print('Pre-trained model snapshot loaded from: %s' % (snapshot_file))
		self.model = self.model.cuda()  # Convert model from CPU to GPU		
		self.model.train()  # Set model to training mode

		self.future_reward_discount = future_reward_discount
		self.criterion = torch.nn.SmoothL1Loss(reduce=False).cuda()  # Initialize Huber loss
		# Initialize optimizer
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=2e-5)
		# self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
		self.iteration = 0

		# Initialize lists to save execution info and RL variables
		self.executed_action_log = []
		self.expected_reward_log = []
		self.current_reward_log = []
		self.predicted_value_log = []

	# Pre-load execution info and RL variables
	def preload(self, transitions_directory):
		self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
		self.iteration = self.executed_action_log.shape[0] - 2
		self.executed_action_log = self.executed_action_log[0:self.iteration, :]
		self.executed_action_log = self.executed_action_log.tolist()
		self.expected_reward_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
		self.expected_reward_log = self.expected_reward_log[0:self.iteration]
		self.expected_reward_log.shape = (self.iteration, 1)
		self.expected_reward_log = self.expected_reward_log.tolist()
		self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'), delimiter=' ')
		self.predicted_value_log = self.predicted_value_log[0:self.iteration]
		self.predicted_value_log.shape = (self.iteration, 1)
		self.predicted_value_log = self.predicted_value_log.tolist()
		self.current_reward_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
		self.current_reward_log = self.current_reward_log[0:self.iteration]
		self.current_reward_log.shape = (self.iteration, 1)
		self.current_reward_log = self.current_reward_log.tolist()

	# Compute for ward pass through model to compute affordances/Q
	def forward(self, state, is_volatile=False, specific_rotation=-1):
		# Pass input data through model
		output_prob = self.model.forward(state, is_volatile, specific_rotation)

		grasp_pred = self._remove_padding(output_prob[0][0])
		for rotate_idx in range(1, len(output_prob)):
			grasp_pred = np.concatenate((grasp_pred,
				self._remove_padding(output_prob[rotate_idx][0])
			), axis=0)
		return grasp_pred

	def _remove_padding(self, prob):
		padding_width, full_width = self.widths
		prob = prob.cpu().data.numpy()
		pred = prob[:, 0,
			(padding_width // 2): (full_width // 2 - padding_width // 2),
			(padding_width // 2): (full_width // 2 - padding_width // 2)
		]
		return pred

	# Compute labels and back-propagate
	def backprop(self, state, action, next_state, current_reward):
		'''
		a Transition
		state: (pre_instruction, prev_color_map, prev_depth_map)
		action: best_pixcel_idx (rotate_idx, x, y)
		next_state:
		rerward:
		'''
		next_grasp_pred = self.forward(next_state, is_volatile=True)
		future_reward = np.max(next_grasp_pred)
		expected_reward = current_reward + self.future_reward_discount * future_reward

		self.expected_reward_log.append([expected_reward])
		self.current_reward_log.append([current_reward])

		label, label_weights = self._get_label_and_weights(action, expected_reward)

		# Compute loss and backward pass
		self.optimizer.zero_grad()
		loss_value = 0
		# Do for-ward pass with specified rotation (to save gradients)
		loss_value += self._backward_loss(state, action[0], label, label_weights)
		# Since grasping is symmetric, train with another for-ward pass of opposite rotation angle
		opposite_rotate_idx = (action[0] + self.model.num_rotations / 2) % self.model.num_rotations
		loss_value += self._backward_loss(state, opposite_rotate_idx, label, label_weights)
		loss_value /= 2

		# print('Training loss: %f' % (loss_value))
		self.optimizer.step()
		return loss_value

	def _backward_loss(self, state, rotate_idx, label, label_weights=None):
		_ = self.forward(state, False, rotate_idx)
		output = self.model.output_prob[0][0].view(1, 320, 320)
		loss = (self.criterion(output, label) * label_weights)
		loss = loss.sum()
		loss.backward()
		return loss.cpu().data.numpy()

	def _get_label_and_weights(self, best_pix_ind, expected_reward):
		# Compute label
		label = np.zeros((1, 320, 320))
		label[0, 48 + best_pix_ind[1], 48 + best_pix_ind[2]] = expected_reward
		label = Variable(torch.from_numpy(label).float().cuda())
		# Compute label mask
		label_weights = np.zeros(label.shape)  # Compute label mask
		label_weights[0, 48 + best_pix_ind[1], 48 + best_pix_ind[2]] = 1
		label_weights = Variable(torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
		return label, label_weights

	def experience_replay(self, reward_value, logger):
		# Do sampling for experience replay
		sample_reward = 0 if reward_value == 1 else 1
		# Get samples of the same primitive but with different results
		sample_idxs = np.argwhere(np.asarray(self.current_reward_log)[1:self.iteration, 0] == sample_reward)

		if sample_idxs.size > 0:
			# Find sample with highest surprise value
			# TODO
			sample_surprise_values = np.abs(
				np.asarray(self.predicted_value_log)[sample_idxs[:, 0]] -
				np.asarray(self.expected_reward_log)[sample_idxs[:, 0]]
			)
			sorted_surprise_idx = np.argsort(sample_surprise_values[:, 0])
			sorted_sample_idxs = sample_idxs[sorted_surprise_idx, 0]
			pow_law_exp = 2
			rand_sample_idxs = int(np.round(np.random.power(pow_law_exp, 1) * (sample_idxs.size - 1)))
			sample_iteration = sorted_sample_idxs[rand_sample_idxs]

			# Load sample instructiont and RGB-D heightmap
			sample_instruction = logger.load_instruction(sample_iteration, '0')
			sample_color_heightmap, sample_depth_heightmap = logger.load_heightmaps(sample_iteration, '0')
			sample_state = (sample_instruction, sample_color_heightmap, sample_depth_heightmap)
			next_instruction = logger.load_instruction(sample_iteration+1, '0')
			next_color_heightmap, next_depth_heightmap = logger.load_heightmaps(sample_iteration+1, '0')
			next_state = (next_instruction, next_color_heightmap, next_depth_heightmap)
			# Get labels for sample and back-propagate
			sample_action = (np.asarray(self.executed_action_log)[sample_iteration, 1:4]).astype(int)
			self.backprop(sample_state, sample_action, next_state, sample_reward)

			# Recompute prediction value, if sample_action == 'grasp':
			sample_grasp_pred = self.forward(sample_state, is_volatile=True)
			self.predicted_value_log[sample_iteration] = [np.max(sample_grasp_pred)]

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
