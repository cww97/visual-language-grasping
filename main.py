#!/usr/bin/env python

import argparse
import threading
import time
import numpy as np
import torch
from trainer import Trainer, Transition
from logger import Logger
import utils
from envs.real.robot import RealRobot
from envs.simulation.robot import SimRobot
from envs.simulation.robot import TestRobot
from utils.config import Config
from tensorboardX import SummaryWriter
from envs.robot import State
from scipy import ndimage


class Solver():

	def __init__(self, args):
		np.random.seed(args.random_seed)  # Set random seed
		self.writer = SummaryWriter()  # tensorboard

		# algorightm options
		self.save_visualizations = args.save_visualizations

		# Initialize pick-and-place system (camera and robot)
		self.workspace_limits = args.workspace_limits
		self.heightmap_resolution = args.heightmap_resolution
		self.robot = SimRobot(args.obj_mesh_dir, args.num_obj, args.workspace_limits)

		# Initialize trainer
		self.snapshot_file = args.snapshot_file
		self.trainer = Trainer(args.future_reward_discount, args.load_snapshot, self.snapshot_file)

		# Initialize data logger
		self.logger = Logger(args.continue_logging, args.logging_directory)
		self.logger.save_camera_info(self.robot.cam_intrinsics, self.robot.cam_pose, self.robot.cam_depth_scale)
		self.logger.save_heightmap_info(args.workspace_limits, self.heightmap_resolution)

		# Find last executed iteration of pre-loaded log, and load execution info and RL variables
		if args.continue_logging:
			self.trainer.preload(self.logger.transitions_directory)

		# between threads(action and training)
		self.executing_action = False  # lock
		self.grasp_pred = None
		self.best_pix = None
		self.color_map = None
		self.valid_depth_heightmap = None
		self.grasp_reward = None

	def _process_actions(self):
		# Parallel thread to process network output and execute actions
		while True:
			if self.executing_action:
				self._get_best_pix()
				best_rotation_angle, primitive_position = self._get_action_data()

				# Initialize variables that influence reward
				self.grasp_reward = self.robot.grasp(
					primitive_position, best_rotation_angle, self.workspace_limits
				).value

				self.executing_action = False  # release the lock
			time.sleep(0.01)

	def _get_best_pix(self):
		best_grasp_conf = np.max(self.grasp_pred)
		print('Primitive confidence scores: %f (grasp)' % (best_grasp_conf))

		# Get pixel location and rotation with highest affordance
		self.best_pix = np.unravel_index(np.argmax(self.grasp_pred), self.grasp_pred.shape)
		predicted_value = np.max(self.grasp_pred)

		# Save predicted confidence value
		self.trainer.predicted_value_log.append([predicted_value])
		self.logger.write_to_log('predicted-value', self.trainer.predicted_value_log)

		# Visualize executed primitive, and affordances
		if self.save_visualizations:
			grasp_pred_vis = self.trainer.get_pred_vis(
				self.grasp_pred, self.color_map, self.best_pix
			)
			self.logger.save_visualizations(self.trainer.iteration, grasp_pred_vis, 'grasp')
			# cv2.imwrite('visualization.grasp.png', grasp_pred_vis)

	def _get_action_data(self):
		# Compute 3D position of pixel
		# print('Action: %s at (%d, %d, %d)' % (
		best_rotation_angle = np.deg2rad(self.best_pix[0] * (360.0 / self.trainer.model.num_rotations))
		best_pix_x = self.best_pix[2]
		best_pix_y = self.best_pix[1]
		primitive_position = [
			best_pix_x * self.heightmap_resolution +
			self.workspace_limits[0][0], best_pix_y * self.heightmap_resolution + self.workspace_limits[1][0],
			self.valid_depth_heightmap[best_pix_y][best_pix_x] + self.workspace_limits[2][0]
		]

		self.trainer.executed_action_log.append([
			1, self.best_pix[0],
			self.best_pix[1],
			self.best_pix[2]
		])  # 1 - grasp
		self.logger.write_to_log('executed-action', self.trainer.executed_action_log)
		return best_rotation_angle, primitive_position

	def main(self):
		action_thread = threading.Thread(target=self._process_actions)
		action_thread.daemon = True
		action_thread.start()

		prev_depth_heightmap = None
		prev_state = None
		prev_action = None
		prev_grasp_reward = None

		# Start main training loop
		self.no_change_cnt = 0
		while True:
			print('\n%s iteration: %d' % ('Training', self.trainer.iteration))
			iteration_time_0 = time.time()
			# Make sure simulation is still stable (if not, reset simulation)
			self.robot.check_sim()
			depth_heightmap = self._get_imgs()
			if self._check_restart(prev_grasp_reward): continue

			print('instruction: %s' % (self.robot.instruction_str))
			cur_state = (self.robot.instruction, self.input_color_data, self.input_depth_data)
			self.grasp_pred = self.trainer.forward(cur_state, is_volatile=True)
			self.executing_action = True  # Robot: Execute action in another thread

			# Run training iteration in current thread (aka training thread)
			if prev_state is not None:
				self._detect_changes(depth_heightmap, prev_depth_heightmap, prev_grasp_reward)
				loss_value = self.trainer.backprop(prev_state, prev_action, cur_state, prev_grasp_reward)
				
				self.trainer.experience_replay(prev_grasp_reward, self.logger)
				self._log_board_save(prev_grasp_reward, loss_value)
				
			# Sync both action thread and training thread
			while self.executing_action: time.sleep(0.01)

			# Save information for next training step
			prev_depth_heightmap = depth_heightmap.copy()
			prev_state = (self.robot.instruction,
						self.input_color_data,
						self.input_depth_data)
			prev_grasp_reward = self.grasp_reward
			prev_action = self.best_pix

			self.trainer.iteration += 1
			iteration_time_1 = time.time()
			print('Time elapsed: %f' % (iteration_time_1 - iteration_time_0))

	def _get_imgs(self):
		# Get latest RGB-D image
		color_img, depth_img = self.robot.get_camera_data()
		depth_img = depth_img * self.robot.cam_depth_scale  # Apply depth scale from calibration

		# Get heightmap from RGB-D image (by re-projecting 3D point cloud)
		self.color_map, depth_heightmap = utils.get_heightmap(
			color_img, depth_img, self.robot.cam_intrinsics,
			self.robot.cam_pose, self.workspace_limits, self.heightmap_resolution
		)
		self.valid_depth_heightmap = depth_heightmap.copy()
		self.valid_depth_heightmap[np.isnan(self.valid_depth_heightmap)] = 0

		# Save RGB-D images and RGB-D heightmaps
		self.logger.save_instruction(self.trainer.iteration, self.robot.instruction_str, '0')
		self.logger.save_images(self.trainer.iteration, color_img, depth_img, '0')
		self.logger.save_heightmaps(self.trainer.iteration, self.color_map, self.valid_depth_heightmap, '0')
		self._preprocess_img(self.color_map, self.valid_depth_heightmap)
		return depth_heightmap

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
		self.input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3, 2, 0, 1)
		self.input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3, 2, 0, 1)

	def _check_restart(self, prev_grasp_reward):
		# Reset simulation or pause real-world training if table is empty
		stuff_count = np.zeros(self.valid_depth_heightmap.shape)
		stuff_count[self.valid_depth_heightmap > 0.02] = 1
		empty_threshold = 300
		workspace_empty = np.sum(stuff_count) < empty_threshold
		if workspace_empty or self.no_change_cnt > 10:
			self.no_change_cnt = 0
			output_state = '%s, reset,' % ('empty' if workspace_empty else 'no change for a long time')
			print(output_state)
			self.robot.restart_sim()
			self.robot.add_objects()

			self.trainer.clearance_log.append([self.trainer.iteration])
			self.logger.write_to_log('clearance', self.trainer.clearance_log)
			return True

		if prev_grasp_reward != 0:  # State.FAIL
			print("grasp {}, restart the simulation".format(
				["success", "wrong"][prev_grasp_reward != State.SUCCESS]
			))
			self.robot.restart_sim()
			self.robot.add_objects()

		return False

	def _detect_changes(self, depth_heightmap, prev_depth_heightmap, prev_grasp_reward):
		# Detect changes
		depth_diff = abs(depth_heightmap - prev_depth_heightmap)
		depth_diff[np.isnan(depth_diff)] = 0
		depth_diff[depth_diff > 0.3] = depth_diff[depth_diff < 0.01] = 0
		depth_diff[depth_diff > 0] = 1
		change_threshold = 300
		change_value = np.sum(depth_diff)
		change_detected = change_value > change_threshold or prev_grasp_reward == 1  # State.SUCCESS
		print('Change detected: %r (value: %d)' % (change_detected, change_value))

		if change_detected:
			self.no_change_cnt = 0
		else:
			self.no_change_cnt += 1

	def _log_board_save(self, prev_grasp_reward, loss_value):
		'''
		$ tensorboard --host 0.0.0.0 --logdir runs
		'''
		self.logger.write_to_log('expected-reward', self.trainer.expected_reward_log)
		self.logger.write_to_log('current-reward', self.trainer.current_reward_log)

		# record on tensorboard
		self.writer.add_scalar('VLG/reward', prev_grasp_reward, self.trainer.iteration)
		self.writer.add_scalar('VLG/loss', loss_value, self.trainer.iteration)

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