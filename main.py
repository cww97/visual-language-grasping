#!/usr/bin/env python

import argparse
import threading
import time
import numpy as np
import torch
from trainer import Trainer, Transition, State
from logger import Logger
import utils
from envs.real.robot import RealRobot
from envs.simulation.robot import SimRobot
from envs.simulation.robot import TestRobot
from utils.config import Config
from tensorboardX import SummaryWriter
from envs.robot import Reward
from scipy import ndimage
from itertools import count


class Solver():

	def __init__(self, args):
		np.random.seed(args.random_seed)  # Set random seed
		self.writer = SummaryWriter()  # tensorboard

		# algorightm options
		self.save_visualizations = args.save_visualizations

		# Initialize pick-and-place system (camera and robot)
		self.workspace_limits = args.workspace_limits
		self.heightmap_resolution = args.heightmap_resolution
		self.env = SimRobot(args.obj_mesh_dir, args.num_obj, args.workspace_limits)
		
		# Initialize trainer
		self.snapshot_file = args.snapshot_file
		self.trainer = Trainer(args.future_reward_discount, args.load_snapshot, self.snapshot_file)

		# Initialize data logger
		self.logger = Logger(args.continue_logging, args.logging_directory)
		self.logger.save_camera_info(self.env.cam_intrinsics, self.env.cam_pose, self.env.cam_depth_scale)
		self.logger.save_heightmap_info(args.workspace_limits, self.heightmap_resolution)
		
		# Find last executed iteration of pre-loaded log, and load execution info and RL variables
		if args.continue_logging:
			self.trainer.preload(self.logger.transitions_directory)

		# between threads(action and training)
		self.color_map = None
		self.valid_depth_heightmap = None

	def _get_best_pix(self, grasp_pred):
		best_grasp_conf = np.max(grasp_pred)
		# print('Primitive confidence scores: %f (grasp)' % (best_grasp_conf))

		# Get pixel location and rotation with highest affordance
		best_pix = np.unravel_index(np.argmax(grasp_pred), grasp_pred.shape)
		predicted_value = np.max(grasp_pred)

		# Visualize executed primitive, and affordances
		if self.save_visualizations:
			grasp_pred_vis = self.trainer.get_pred_vis(grasp_pred, self.color_map, best_pix)
			self.logger.save_visualizations(self.trainer.iteration, grasp_pred_vis, 'grasp')
			# cv2.imwrite('visualization.grasp.png', grasp_pred_vis)
		return best_pix

	def _get_action_data(self, best_pix):
		# Compute 3D position of pixel
		best_rotation_angle = np.deg2rad(best_pix[0] * (360.0 / self.trainer.model.num_rotations))
		best_pix_x = best_pix[2]
		best_pix_y = best_pix[1]
		primitive_position = [
			best_pix_x * self.heightmap_resolution + self.workspace_limits[0][0], 
			best_pix_y * self.heightmap_resolution + self.workspace_limits[1][0],
			self.valid_depth_heightmap[best_pix_y][best_pix_x] + self.workspace_limits[2][0]
		]
		self.trainer.action_log.append([1, best_pix[0], best_pix[1], best_pix[2]])  # 1 - grasp
		self.logger.write_to_log('action', self.trainer.action_log)
		return best_rotation_angle, primitive_position

	def main(self):
		optim_thread = threading.Thread(target=self._optimize_model)
		optim_thread.daemon = True
		optim_thread.start()

		for epoch in count():  # Start main training loop
			self.env.reset()
			# print('instruction: %s' % (self.env.instruction_str))
			self.no_change_cnt = 0
			state = State(self.env.instruction, *self._get_imgs())
			for t in count():
				time_0 = time.time()

				grasp_pred = self.trainer.forward(state, is_volatile=True)
				action = self._get_best_pix(grasp_pred)
				angle, position = self._get_action_data(action)
				reward, done = self.env.step(position, angle, self.workspace_limits)
				next_state = State(self.env.instruction, *self._get_imgs())  # observe new state
				self.trainer.memory.push(state, action, next_state, reward)

				state = next_state
				self._log_board_save(reward)
				print('Iter: %d, Reward = %d, Time: %f' % (self.trainer.iteration, reward, (time.time() - time_0)))
				self.trainer.iteration += 1

				self._detect_changes(next_state.depth_data, state.depth_data, reward)
				if done or self._check_stupid() or (not self.env.is_stable()): break
			# import pdb; pdb.set_trace()

	def _optimize_model(self):
		while True:
			if self.trainer.iteration % self.trainer.BATCH_SIZE == 0:
				loss = self.trainer.optimize_model()
				if loss == None: continue
				self.writer.add_scalar('VLG/loss', loss, self.trainer.iteration)
				print("model updated, loss = %.4f" % (loss))
			time.sleep(1)

	def _get_imgs(self):
		# Get latest RGB-D image
		color_img, depth_img = self.env.get_camera_data()
		depth_img = depth_img * self.env.cam_depth_scale  # Apply depth scale from calibration

		# Get heightmap from RGB-D image (by re-projecting 3D point cloud)
		self.color_map, depth_heightmap = utils.get_heightmap(
			color_img, depth_img, self.env.cam_intrinsics,
			self.env.cam_pose, self.workspace_limits, self.heightmap_resolution
		)
		self.valid_depth_heightmap = depth_heightmap.copy()
		self.valid_depth_heightmap[np.isnan(self.valid_depth_heightmap)] = 0

		# Save RGB-D images and RGB-D heightmaps
		self.logger.save_instruction(self.trainer.iteration, self.env.instruction_str, '0')
		self.logger.save_images(self.trainer.iteration, color_img, depth_img, '0')
		self.logger.save_heightmaps(self.trainer.iteration, self.color_map, self.valid_depth_heightmap, '0')
		input_color_data, input_depth_data = self._preprocess_img(self.color_map, self.valid_depth_heightmap)
		return input_color_data, input_depth_data

	def _preprocess_img(self, color_heightmap, depth_heightmap):
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
		_shape = input_color_image.shape
		assert input_color_image.shape == input_depth_image.shape
		input_color_image.shape = (_shape[0], _shape[1], _shape[2], 1)
		input_depth_image.shape = (_shape[0], _shape[1], _shape[2], 1)
		self.trainer.widths = (padding_width, color_heightmap_2x.shape[0])
		input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3, 2, 0, 1)
		input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3, 2, 0, 1)
		return input_color_data, input_depth_data

	def _check_stupid(self):
		if self.no_change_cnt > 10:
			self.no_change_cnt = 0
			print('no change for a long time, Reset.')
			return True
		return False

	def _detect_changes(self, next_depth_data, depth_data, reward):
		# Detect changes
		depth_diff = abs(next_depth_data - depth_data)
		depth_diff[np.isnan(depth_diff)] = 0
		depth_diff[depth_diff > 0.3] = depth_diff[depth_diff < 0.01] = 0
		depth_diff[depth_diff > 0] = 1
		change_threshold = 3000  # TODO this value, 300
		# import pdb; pdb.set_trace()
		change_value = int(torch.sum(depth_diff))  # np.sum
		change_detected = change_value > change_threshold or reward == 1  # State.SUCCESS
		# print('Change detected: %r (value: %d)' % (change_detected, change_value))

		if change_detected:
			self.no_change_cnt = 0
		else:
			self.no_change_cnt += 1

	def _log_board_save(self, reward):
		'''
		$ tensorboard --host 0.0.0.0 --logdir runs
		'''
		self.logger.write_to_log('reward', self.trainer.reward_log)
		self.writer.add_scalar('VLG/reward', reward, self.trainer.iteration)

		# Save model snapshot
		self.logger.save_backup_model(self.trainer.model, 'reinforcement')
		if self.trainer.iteration % 50 == 0:
			self.logger.save_model(self.trainer.iteration, self.trainer.model, 'reinforcement')
			self.trainer.model = self.trainer.model.cuda()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Train robotic agents to learn visual language grasp.'
	)
	# Run main program with specified config file
	parser.add_argument('-f', '--file', dest='file')
	args = parser.parse_args()

	solver = Solver(Config(args.file))
	solver.main()