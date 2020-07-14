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
		self.env = SimRobot(args.obj_mesh_dir, args.num_obj, args.workspace_limits, args.heightmap_resolution)
		
		# Initialize trainer
		self.snapshot_file = args.snapshot_file
		self.trainer = Trainer(args.future_reward_discount, args.load_snapshot, self.snapshot_file)
		self.env_step_args = (self.trainer.model.num_rotations, self.heightmap_resolution)

		# Initialize data logger
		self.logger = Logger(args.continue_logging, args.logging_directory)
		self.logger.save_camera_info(self.env.cam_intrinsics, self.env.cam_pose, self.env.cam_depth_scale)
		self.logger.save_heightmap_info(args.workspace_limits, self.heightmap_resolution)
		if args.continue_logging:
			self.trainer.preload(self.logger.transitions_directory)

	def main(self):
		# optim_thread = threading.Thread(target=self._optimize_model)
		# optim_thread.daemon = True
		# optim_thread.start()

		for epoch in count():  # Start main training loop
			self.env.reset()
			# print('instruction: %s' % (self.env.instruction_str))
			self.no_change_cnt = 0
			state = State(self.env.instruction, *self._get_imgs())
			for t in count():
				time_0 = time.time()
				action = self.trainer.select_action(state, is_volatile=True, env=self.env, logger=self.logger)
				reward, done = self.env.step(action, state.depth_map, *self.env_step_args)
				next_state = State(self.env.instruction, *self._get_imgs())  # observe new state
				self.trainer.memory.push(state, action, next_state, reward)

				state = next_state
				self._log_board_save(reward, time_0)
				self._detect_changes(next_state.depth_map, state.depth_map, reward)
				if done or self._check_stupid() or (not self.env.is_stable()):
					break
			loss = self.trainer.optimize_model()
			if loss:
				self.writer.add_scalar('VLG/loss', loss, self.trainer.iteration)
			if epoch % 5 == 0:
				self.trainer.target_net.load_state_dict(self.trainer.model.state_dict())

	def _optimize_model(self):
		TARGET_UPDATE = 5
		while True:
			if self.trainer.iteration % (self.trainer.BATCH_SIZE / 2) == 0:
				loss = self.trainer.optimize_model()
				if loss == None: continue
				self.writer.add_scalar('VLG/loss', loss, self.trainer.iteration)
			if self.trainer.iteration % (self.trainer.BATCH_SIZE * TARGET_UPDATE) == 0:
				self.trainer.target_net.load_state_dict(self.trainer.model.state_dict())
			time.sleep(1)

	def _get_imgs(self):
		# Get latest RGB-D image
		color_img, depth_img = self.env.get_camera_data()
		depth_img = depth_img * self.env.cam_depth_scale

		# Get heightmap from RGB-D image (by re-projecting 3D point cloud)
		color_map, depth_heightmap = utils.get_heightmap(
			color_img, depth_img, self.env.cam_intrinsics,
			self.env.cam_pose, self.workspace_limits, self.heightmap_resolution
		)
		# valid_depth_heightmap = depth_heightmap
		depth_heightmap[np.isnan(depth_heightmap)] = 0

		# Save RGB-D images and RGB-D heightmaps
		self.logger.save_instruction(self.trainer.iteration, self.env.instruction_str, '0')
		self.logger.save_images(self.trainer.iteration, color_img, depth_img, '0')
		self.logger.save_heightmaps(self.trainer.iteration, color_map, depth_heightmap, '0')
		return color_map, depth_heightmap

	def _check_stupid(self):
		if self.no_change_cnt > 10:
			self.no_change_cnt = 0
			# print('no change for a long time, Reset.')
			return True
		return False

	def _detect_changes(self, next_depth_data, depth_data, reward):
		# Detect changes
		depth_diff = abs(next_depth_data - depth_data)
		depth_diff[np.isnan(depth_diff)] = 0
		depth_diff[depth_diff > 0.3] = depth_diff[depth_diff < 0.01] = 0
		depth_diff[depth_diff > 0] = 1
		change_threshold = 300  # TODO this value, 300
		change_value = np.sum(depth_diff)  # np.sum
		change_detected = change_value > change_threshold or reward == 1  # State.SUCCESS
		# print('Change detected: %r (value: %d)' % (change_detected, change_value))

		if change_detected:
			self.no_change_cnt = 0
		else:
			self.no_change_cnt += 1

	def _log_board_save(self, reward, time_0):
		'''
		$ tensorboard --host 0.0.0.0 --logdir runs
		'''
		self.trainer.reward_log.append([reward])
		self.logger.write_to_log('reward', self.trainer.reward_log)
		self.writer.add_scalar('VLG/reward', reward, self.trainer.iteration)

		# Save model snapshot
		self.logger.save_backup_model(self.trainer.model, 'reinforcement')
		if self.trainer.iteration % 50 == 0:
			self.logger.save_model(self.trainer.iteration, self.trainer.model, 'reinforcement')
			self.trainer.model = self.trainer.model.cuda()
		print('Iter: %d, Reward = %d, Time: %.2f' % (self.trainer.iteration, reward, (time.time() - time_0)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Train robotic agents to learn visual language grasp.'
	)
	# Run main program with specified config file
	parser.add_argument('-f', '--file', dest='file')
	args = parser.parse_args()
	solver = Solver(Config(args.file))
	'''
	angles = []
	for i in range(8):
		angle = np.deg2rad(i * (360.0 / 16))
		tool_rotation_angle = (angle % np.pi) - np.pi / 2
		angles.append(tool_rotation_angle)
	solver.env.reset()
	solver.env.random_grasp_action()
	assert False
	'''
	solver.main()