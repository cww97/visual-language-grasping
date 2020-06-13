import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.autograd import Variable

from envs.data import Data as TextData
from envs.robot import State
from models import reactive_net, reinforcement_net
from utils import CrossEntropyLoss2d
# from envs.data import Data as TextData
# import pdb


class Trainer(object):

	def __init__(self, method, future_reward_discount,
				 is_testing, load_snapshot, snapshot_file):
		assert torch.cuda.is_available()
		self.method = method
		self.text_data = TextData()
		vocab, pad_idx = len(self.text_data.text_field.vocab), self.text_data.padding_idx

		# Fully convolutional classification network for supervised learning
		if self.method == 'reactive':
			self.model = reactive_net(vocab_size=vocab, padding_idx=pad_idx)

			# Initialize classification loss
			grasp_num_classes = 3  # 0 - grasp, 1 - failed grasp, 2 - no loss
			grasp_class_weights = torch.ones(grasp_num_classes)
			grasp_class_weights[grasp_num_classes - 1] = 0
			self.grasp_criterion = CrossEntropyLoss2d(grasp_class_weights.cuda()).cuda()

		# Fully convolutional Q network for deep reinforcement learning
		elif self.method == 'reinforcement':
			self.model = reinforcement_net(vocab_size=vocab, padding_idx=pad_idx)
			self.future_reward_discount = future_reward_discount
			# Initialize Huber loss
			self.criterion = torch.nn.SmoothL1Loss(reduce=False).cuda()  # Huber losss

		# Load pre-trained model
		if load_snapshot:
			self.model.load_state_dict(torch.load(snapshot_file))
			print('Pre-trained model snapshot loaded from: %s' % (snapshot_file))

		# Convert model from CPU to GPU
		self.model = self.model.cuda()

		# Set model to training mode
		self.model.train()

		# Initialize optimizer
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=2e-5)
		# self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
		self.iteration = 0

		# Initialize lists to save execution info and RL variables
		self.executed_action_log = []
		self.label_value_log = []
		self.reward_value_log = []
		self.predicted_value_log = []
		self.use_heuristic_log = []
		self.clearance_log = []
		self.grasp_success_log = []

	# Pre-load execution info and RL variables
	def preload(self, transitions_directory):
		self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
		self.iteration = self.executed_action_log.shape[0] - 2
		self.executed_action_log = self.executed_action_log[0:self.iteration, :]
		self.executed_action_log = self.executed_action_log.tolist()
		self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
		self.label_value_log = self.label_value_log[0:self.iteration]
		self.label_value_log.shape = (self.iteration, 1)
		self.label_value_log = self.label_value_log.tolist()
		self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'), delimiter=' ')
		self.predicted_value_log = self.predicted_value_log[0:self.iteration]
		self.predicted_value_log.shape = (self.iteration, 1)
		self.predicted_value_log = self.predicted_value_log.tolist()
		self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
		self.reward_value_log = self.reward_value_log[0:self.iteration]
		self.reward_value_log.shape = (self.iteration, 1)
		self.reward_value_log = self.reward_value_log.tolist()
		self.use_heuristic_log = np.loadtxt(os.path.join(transitions_directory, 'use-heuristic.log.txt'), delimiter=' ')
		self.use_heuristic_log = self.use_heuristic_log[0:self.iteration]
		self.use_heuristic_log.shape = (self.iteration, 1)
		self.use_heuristic_log = self.use_heuristic_log.tolist()
		# self.clearance_log = np.loadtxt(os.path.join(transitions_directory, 'clearance.log.txt'), delimiter=' ')
		# self.clearance_log.shape = (self.clearance_log.shape[0], 1)
		# self.clearance_log = self.clearance_log.tolist()

		self.grasp_success_log = np.loadtxt(os.path.join(transitions_directory, 'grasp-success.log.txt'), delimiter=' ')
		self.grasp_success_log = self.grasp_success_log[0:self.iteration]
		self.grasp_success_log.shape = (self.iteration, 1)
		self.grasp_success_log = self.grasp_success_log.tolist()
		

	# Compute for ward pass through model to compute affordances/Q
	def forward(self, instruction, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=-1):

		# cslnb, convert text_instruction -> text_tensor
		instruction_tensor = self.text_data.get_tensor(instruction).cuda()

		# Apply 2x scale to input heightmaps
		color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2, 2, 1], order=0)
		depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2, 2], order=0)
		assert(color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])

		# Add extra padding (to handle rotations inside network)
		diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
		diag_length = np.ceil(diag_length / 32) * 32
		padding_width = int((diag_length - color_heightmap_2x.shape[0]) / 2)
		color_heightmap_2x_r = np.pad(color_heightmap_2x[:, :, 0], padding_width, 'constant', constant_values=0)
		color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
		color_heightmap_2x_g = np.pad(color_heightmap_2x[:, :, 1], padding_width, 'constant', constant_values=0)
		color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
		color_heightmap_2x_b = np.pad(color_heightmap_2x[:, :, 2], padding_width, 'constant', constant_values=0)
		color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
		color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
		depth_heightmap_2x = np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)

		# Pre-process color image (scale and normalize)
		image_mean = [0.485, 0.456, 0.406]
		image_std = [0.229, 0.224, 0.225]
		input_color_image = color_heightmap_2x.astype(float) / 255
		for c in range(3):
			input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c]) / image_std[c]

		# Pre-process depth image (normalize)
		image_mean = [0.01, 0.01, 0.01]
		image_std = [0.03, 0.03, 0.03]
		depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
		input_depth_image = np.concatenate((depth_heightmap_2x, depth_heightmap_2x, depth_heightmap_2x), axis=2)
		for c in range(3):
			input_depth_image[:, :, c] = (input_depth_image[:, :, c] - image_mean[c]) / image_std[c]

		# Construct minibatch of size 1 (b,c,h,w)
		input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
		input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
		input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3, 2, 0, 1)
		input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3, 2, 0, 1)

		# Pass input data through model, cslnb, !!!!!!!!!!!!!!!!!!!!!!!!!!!
		output_prob, state_feat = self.model.forward(
			instruction_tensor, input_color_data, input_depth_data, is_volatile, specific_rotation
		)

		full_width = color_heightmap_2x.shape[0]
		grasp_predictions = self._get_pred(output_prob[0][0], padding_width, full_width)
		for rotate_idx in range(1, len(output_prob)):
			grasp_predictions = np.concatenate((
				grasp_predictions,
				self._get_pred(output_prob[rotate_idx][0], padding_width, full_width)
			), axis=0)
		# import pdb; pdb.set_trace()
		return grasp_predictions, state_feat

	def _get_pred(self, prob, padding_width, full_width):
		if self.method == 'reactive':
			prob = F.softmax(prob, dim=1)  # Return affordances
		# else (reinforcement): Return Q values
		prob = prob.cpu().data.numpy()
		pred = prob[
			:, 0,
			(padding_width // 2): (full_width // 2 - padding_width // 2),
			(padding_width // 2): (full_width // 2 - padding_width // 2)
		]
		return pred

	def get_label_value(
		self,
		grasp_success,
		change_detected,
		prev_grasp_predictions,
		instruction,
		next_color_heightmap,
		next_depth_heightmap
	):
		if self.method == 'reactive':
			# label: 0 - grasp, 1 - failed grasp, 2 - no loss
			label_value = 0 if grasp_success == State.SUCCESS else 1  # TODO, wrong obj
			print('grasp_success = %s, Label value: %d' % (str(grasp_success), label_value))
			label_value, reward_value = label_value, label_value

		elif self.method == 'reinforcement':
			# Compute current reward, deal with put in the future
			current_reward = 1.0 if grasp_success == State.SUCCESS else 0.0

			# Compute future reward
			if not change_detected and not grasp_success == State.SUCCESS:
				future_reward = 0
			else:
				next_grasp_predictions, next_state_feat = self.forward(
					instruction, next_color_heightmap, next_depth_heightmap, is_volatile=True
				)
				future_reward = np.max(next_grasp_predictions)
			print('Reward: (Current = %f, Future = %f)' % (current_reward, future_reward))
			expected_reward = current_reward + self.future_reward_discount * future_reward
			print('Expected reward: %f + %f x %f = %f' % (
				current_reward, self.future_reward_discount, future_reward, expected_reward
			))
			label_value, reward_value = expected_reward, current_reward

		self.label_value_log.append([label_value])
		self.reward_value_log.append([reward_value])
		self.grasp_success_log.append([grasp_success.value])
		return label_value, reward_value

	# Compute labels and back-propagate
	def backprop(self, instruction, color_heightmap, depth_heightmap, best_pix_ind, label_value):
		fill_value = 2 if self.method == 'reactive' else 0  # Compute fill value, 2:no loss
		# Compute labels
		label = np.zeros((1, 320, 320)) + fill_value
		label[0, 48 + best_pix_ind[1], 48 + best_pix_ind[2]] = label_value
		# Compute label mask
		if self.method == 'reactive': label_weights = None  # just a fill
		elif self.method == 'reinforcement':
			label_weights = np.zeros(label.shape)  # Compute label mask
			label_weights[0, 48 + best_pix_ind[1], 48 + best_pix_ind[2]] = 1

		# Compute loss and backward pass
		self.optimizer.zero_grad()
		loss_value = 0
		# if primitive_action == 'grasp':
		# Do for-ward pass with specified rotation (to save gradients)
		loss_value += self._backward_loss(instruction, color_heightmap, depth_heightmap, best_pix_ind[0],
										  label, label_weights)
		# Since grasping is symmetric, train with another for-ward pass of opposite rotation angle
		opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations / 2) % self.model.num_rotations
		loss_value += self._backward_loss(instruction, color_heightmap, depth_heightmap,
										  opposite_rotate_idx, label, label_weights)
		loss_value = loss_value / 2

		print('Training loss: %f' % (loss_value))
		self.optimizer.step()
		return loss_value

	def _backward_loss(self, instruction, color_heightmap, depth_heightmap, rotate_idx, label, label_weights=None):
		_, _ = self.forward(instruction, color_heightmap, depth_heightmap, False, rotate_idx)
		if self.method == 'reactive':
			loss = self.grasp_criterion(
				self.model.output_prob[0][0],
				Variable(torch.from_numpy(label).long().cuda())
			)
		else:
			loss = self.criterion(
				self.model.output_prob[0][0].view(1, 320, 320),
				Variable(torch.from_numpy(label).float().cuda())
			) * Variable(torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
			loss = loss.sum()
		# import pdb; pdb.set_trace()
		loss.backward()
		return loss.cpu().data.numpy()

	def experience_replay(self, reward_value, logger):
		# Do sampling for experience replay
		sample_reward_value = 0 if reward_value == 1 else 1  # jingsui
		# Get samples of the same primitive but with different results
		sample_idxs = np.argwhere(
			np.asarray(self.reward_value_log)[1:self.iteration, 0] == sample_reward_value
		)

		if sample_idxs.size > 0:
			# Find sample with highest surprise value
			if self.method == 'reactive':
				sample_surprise_values = np.abs(
					np.asarray(self.predicted_value_log)[sample_idxs[:, 0]] - (1 - sample_reward_value)
				)
			elif self.method == 'reinforcement':
				sample_surprise_values = np.abs(
					np.asarray(self.predicted_value_log)[sample_idxs[:, 0]] -
					np.asarray(self.label_value_log)[sample_idxs[:, 0]]
				)
			sorted_surprise_idx = np.argsort(sample_surprise_values[:, 0])
			sorted_sample_idxs = sample_idxs[sorted_surprise_idx, 0]
			pow_law_exp = 2
			rand_sample_idxs = int(np.round(np.random.power(pow_law_exp, 1) * (sample_idxs.size - 1)))
			sample_iteration = sorted_sample_idxs[rand_sample_idxs]
			print(
				'Experience replay: iteration %d (surprise value: %f)' %
				(sample_iteration, sample_surprise_values[sorted_surprise_idx[rand_sample_idxs]])
			)

			# Load sample instructiont and RGB-D heightmap
			# logger.get_date(sample_iteration)
			sample_instruction = logger.load_instruction(sample_iteration, '0')
			sample_color_heightmap, sample_depth_heightmap = logger.load_heightmaps(sample_iteration, '0')

			# Compute For-ward pass with sample
			sample_grasp_predictions, sample_state_feat = self.forward(
				sample_instruction, sample_color_heightmap, sample_depth_heightmap, is_volatile=True
			)

			# Get labels for sample and back-propagate
			sample_pix_idx = (np.asarray(self.executed_action_log)[sample_iteration, 1:4]).astype(int)
			self.backprop(
				sample_instruction, sample_color_heightmap, sample_depth_heightmap,
				sample_pix_idx, sample_reward_value
			)

			# Recompute prediction value, if sample_action == 'grasp':
			self.predicted_value_log[sample_iteration] = [np.max(sample_grasp_predictions)]

	def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind):
		canvas = None
		num_rotations = predictions.shape[0]
		for canvas_row in range(int(num_rotations / 4)):
			tmp_row_canvas = None
			for canvas_col in range(4):
				rotate_idx = canvas_row * 4 + canvas_col
				prediction_vis = predictions[rotate_idx, :, :].copy()
				# prediction_vis[prediction_vis < 0] = 0 # assume probability
				# prediction_vis[prediction_vis > 1] = 1 # assume probability
				prediction_vis = np.clip(prediction_vis, 0, 1)
				prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
				prediction_vis = cv2.applyColorMap((prediction_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
				if rotate_idx == best_pix_ind[0]:
					prediction_vis = cv2.circle(
						prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0, 0, 255), 2
					)
				prediction_vis = ndimage.rotate(prediction_vis, rotate_idx * (360.0 / num_rotations), reshape=False, order=0)
				background_image = ndimage.rotate(color_heightmap, rotate_idx * (360.0 / num_rotations), reshape=False, order=0)
				prediction_vis = (0.5 * cv2.cvtColor(background_image,
								  cv2.COLOR_RGB2BGR) + 0.5 * prediction_vis).astype(np.uint8)
				if tmp_row_canvas is None:
					tmp_row_canvas = prediction_vis
				else:
					tmp_row_canvas = np.concatenate((tmp_row_canvas, prediction_vis), axis=1)
			if canvas is None:
				canvas = tmp_row_canvas
			else:
				canvas = np.concatenate((canvas, tmp_row_canvas), axis=0)

		return canvas

	def grasp_heuristic(self, depth_heightmap):

		num_rotations = 16

		for rotate_idx in range(num_rotations):
			rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx * (360.0 / num_rotations), reshape=False, order=0)
			valid_areas = np.zeros(rotated_heightmap.shape)
			valid_areas[np.logical_and(
				rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0, -25], order=0) > 0.02,
				rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0, 25], order=0) > 0.02
			)] = 1
			# valid_areas = np.multiply(valid_areas, rotated_heightmap)
			blur_kernel = np.ones((25, 25), np.float32) / 9
			valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
			tmp_grasp_predictions = ndimage.rotate(valid_areas, -rotate_idx * (360.0 / num_rotations), reshape=False, order=0)
			tmp_grasp_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

			if rotate_idx == 0:
				grasp_predictions = tmp_grasp_predictions
			else:
				grasp_predictions = np.concatenate((grasp_predictions, tmp_grasp_predictions), axis=0)

		best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
		return best_pix_ind


if __name__ == '__main__':
	loaded_snapshot_state_dict = torch.load('downloads/vpg-original-real-pretrained-30-obj.pth')
	with open('model_dict.txt', 'w') as f:
		f.write(str(loaded_snapshot_state_dict.keys()))
