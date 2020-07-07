#!/usr/bin/env python

import argparse
import threading
import time
import numpy as np
import torch
from trainer import Trainer
from logger import Logger
import utils
from envs.real.robot import RealRobot
from envs.simulation.robot import SimRobot
from envs.simulation.robot import TestRobot
from utils.config import Config
from tensorboardX import SummaryWriter
from envs.robot import State


class Solver():

	def __init__(self, args):
		np.random.seed(args.random_seed)  # Set random seed
		self.writer = SummaryWriter()  # tensorboard

		# test option
		self.is_testing = args.is_testing
		self.max_test_trials = args.max_test_trials

		# algorightm options
		self.method = args.method
		self.experience_replay = args.experience_replay
		self.heuristic_bootstrap = args.heuristic_bootstrap
		self.save_visualizations = args.save_visualizations

		# Initialize pick-and-place system (camera and robot)
		self.is_sim = args.is_sim
		self.workspace_limits = args.workspace_limits
		self.heightmap_resolution = args.heightmap_resolution
		if self.is_sim:
			if args.test_preset_file:
				self.robot = TestRobot(args.obj_mesh_dir, args.num_obj,
									   args.workspace_limits, args.test_preset_file)
			else:
				self.robot = SimRobot(args.obj_mesh_dir, args.num_obj, args.workspace_limits)
		else:
			self.robot = RealRobot(args.tcp_host_ip, args.tcp_port, args.rtc_host_ip, args.rtc_port, args.workspace_limits)

		# Initialize trainer
		self.snapshot_file = args.snapshot_file
		self.trainer = Trainer(args.future_reward_discount, self.is_testing, args.load_snapshot, self.snapshot_file)

		# Initialize data logger
		self.logger = Logger(args.continue_logging, args.logging_directory)
		# Save camera intrinsics and pose
		self.logger.save_camera_info(self.robot.cam_intrinsics, self.robot.cam_pose, self.robot.cam_depth_scale)
		# Save heightmap parameters
		self.logger.save_heightmap_info(args.workspace_limits, self.heightmap_resolution)

		# Find last executed iteration of pre-loaded log, and load execution info and RL variables
		if args.continue_logging:
			self.trainer.preload(self.logger.transitions_directory)
		# Initialize variables for heuristic bootstrapping and exploration probability
		self.no_change_cnt = 2 if not args.is_testing else 0

		# between threads(action and training)
		self.executing_action = False  # lock
		self.grasp_pred = None
		self.best_pix = None
		self.color_map = None
		self.valid_depth_heightmap = None
		self.grasp_success = None

	def _process_actions(self):
		# Parallel thread to process network output and execute actions
		while True:
			if self.executing_action:
				self._get_best_pix()
				best_rotation_angle, primitive_position = self._get_action_data()

				# Initialize variables that influence reward
				self.grasp_success = self.robot.grasp(  # Execute
					primitive_position, best_rotation_angle, self.workspace_limits
				)

				self.executing_action = False  # release the lock
			time.sleep(0.01)

	def _get_best_pix(self):
		best_grasp_conf = np.max(self.grasp_pred)
		print('Primitive confidence scores: %f (grasp)' % (best_grasp_conf))

		# If heuristic bootstrapping is enabled: if change has not been detected more than 2 times,
		# execute heuristic algorithm to detect grasps
		# NOTE: typically not necessary and can reduce final performance.
		if self.heuristic_bootstrap and self.no_change_cnt >= 5:
			print('Change not detected for more than 2 grasps. Run **heuristic grasping**.')
			self.best_pix = self.trainer.grasp_heuristic(self.valid_depth_heightmap)
			# self.no_change_cnt = 0
			# import pdb; pdb.set_trace()
			predicted_value = self.grasp_pred[self.best_pix]
			use_heuristic = True
		else:
			use_heuristic = False

			# Get pixel location and rotation with highest affordance
			# prediction from heuristic algorithms (rotation, y, x)
			self.best_pix = np.unravel_index(np.argmax(self.grasp_pred), self.grasp_pred.shape)
			predicted_value = np.max(self.grasp_pred)

		self.trainer.use_heuristic_log.append([1 if use_heuristic else 0])
		self.logger.write_to_log('use-heuristic', self.trainer.use_heuristic_log)

		# Save predicted confidence value
		self.trainer.predicted_value_log.append([predicted_value])
		self.logger.write_to_log('predicted-value', self.trainer.predicted_value_log)

		# Visualize executed primitive, and affordances
		if self.save_visualizations:
			grasp_pred_vis = self.trainer.get_pred_vis(
				self.grasp_pred, self.color_map, self.best_pix
			)
			self.logger.save_visualizations(self.trainer.iteration, grasp_pred_vis, 'grasp')

			atten_vis = self.trainer.get_atten_vis(self.attens, self.color_map, self.best_pix)
			self.logger.save_visualizations(self.trainer.iteration, atten_vis, 'atten')
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
		self.exit_called = False

		prev_color_map = None
		prev_depth_heightmap = None
		prev_depth_map = None
		prev_grasp_success = None
		prev_grasp_pred = None
		prev_best_pix_idx = None

		# Start main training/testing loop
		while True:
			print('\n%s iteration: %d' % ('Testing' if self.is_testing else 'Training', self.trainer.iteration))
			iteration_time_0 = time.time()

			# Make sure simulation is still stable (if not, reset simulation)
			if self.is_sim: self.robot.check_sim()
			depth_heightmap = self._get_imgs()
			if self._check_restart(prev_grasp_success): continue

			if not self.exit_called:  # Run For-ward pass with network to get affordances
				print('instruction: %s' % (self.robot.instruction))  # nb
				self.grasp_pred, _, self.attens = self.trainer.forward(
					self.robot.instruction, self.color_map, self.valid_depth_heightmap, is_volatile=True
				)
				self.executing_action = True  # Robot: Execute action in another thread

			# Run training iteration in current thread (aka training thread)
			if prev_color_map is not None:
				change_detected = self._detect_changes(depth_heightmap, prev_depth_heightmap, prev_grasp_success)

				# Compute training labels
				expected_reward, current_reward = self.trainer.get_reward_value(
					prev_grasp_success, change_detected, prev_grasp_pred,
					self.robot.instruction, self.color_map, self.valid_depth_heightmap
				)

				# Back-propagate
				pre_env = (self.robot.instruction, prev_color_map, prev_depth_map)
				loss_value = self.trainer.backprop(pre_env, prev_best_pix_idx, expected_reward)

				if self.experience_replay and not self.is_testing:
					self.trainer.experience_replay(current_reward, self.logger)

				self._log_board_save(prev_grasp_success, expected_reward, current_reward, loss_value)
				
			# Sync both action thread and training thread
			while self.executing_action: time.sleep(0.01)
			if self.exit_called: break

			# Save information for next training step
			prev_color_map = self.color_map.copy()
			prev_depth_heightmap = depth_heightmap.copy()
			prev_depth_map = self.valid_depth_heightmap.copy()
			prev_grasp_success = self.grasp_success
			prev_grasp_pred = self.grasp_pred.copy()
			prev_best_pix_idx = self.best_pix

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
		self.logger.save_instruction(self.trainer.iteration, self.robot.instruction, '0')
		self.logger.save_images(self.trainer.iteration, color_img, depth_img, '0')
		self.logger.save_heightmaps(self.trainer.iteration, self.color_map, self.valid_depth_heightmap, '0')

		return depth_heightmap

	def _check_restart(self, prev_grasp_success):
		# Reset simulation or pause real-world training if table is empty
		stuff_count = np.zeros(self.valid_depth_heightmap.shape)
		stuff_count[self.valid_depth_heightmap > 0.02] = 1
		empty_threshold = 300 if not (self.is_sim and self.is_testing) else 10
		# TODO
		# if self.is_sim and self.is_testing: empty_threshold = 10
		workspace_empty = np.sum(stuff_count) < empty_threshold
		if workspace_empty or (self.is_sim and self.no_change_cnt > 10):
			self.no_change_cnt = 0
			if self.is_sim:
				output_state = '%s, reset,' % ('empty' if workspace_empty else 'no change for a long time')
				print(output_state)
				self.robot.restart_sim()
				self.robot.add_objects()
				if self.is_testing:  # If at end of test run, re-load original weights (before test run)
					self.trainer.model.load_state_dict(torch.load(self.snapshot_file))
			else:
				print('Not enough stuff on the table (value: %d)! Flipping over bin of objects...' % (np.sum(stuff_count)))
				self.robot.restart_real()

			self.trainer.clearance_log.append([self.trainer.iteration])
			self.logger.write_to_log('clearance', self.trainer.clearance_log)
			if self.is_testing and len(self.trainer.clearance_log) >= self.max_test_trials:
				self.exit_called = True  # Exit after training thread (back-prop and saving labels)
			return True

		if self.is_sim and prev_grasp_success != State.FAIL:
			print("grasp {}, restart the simulation".format(
				["success", "wrong"][prev_grasp_success != State.SUCCESS]
			))
			self.robot.restart_sim()
			self.robot.add_objects()

		return False

	def _detect_changes(self, depth_heightmap, prev_depth_heightmap, prev_grasp_success):
		# Detect changes
		depth_diff = abs(depth_heightmap - prev_depth_heightmap)
		depth_diff[np.isnan(depth_diff)] = 0
		depth_diff[depth_diff > 0.3] = depth_diff[depth_diff < 0.01] = 0
		depth_diff[depth_diff > 0] = 1
		change_threshold = 300
		change_value = np.sum(depth_diff)
		change_detected = change_value > change_threshold or prev_grasp_success == State.SUCCESS
		print('Change detected: %r (value: %d)' % (change_detected, change_value))

		if change_detected:
			self.no_change_cnt = 0
		else:
			self.no_change_cnt += 1

		return change_detected

	def _log_board_save(self, prev_grasp_success, expected_reward, current_reward, loss_value):
		'''
		$ tensorboard --host 0.0.0.0 --logdir runs
		'''
		self.logger.write_to_log('expected-reward', self.trainer.expected_reward_log)
		self.logger.write_to_log('current-reward', self.trainer.current_reward_log)

		# record on tensorboard
		self.writer.add_scalar('VLG/grasp_success', prev_grasp_success.value, self.trainer.iteration)
		self.writer.add_scalar('VLG/expected_reward', expected_reward, self.trainer.iteration)
		self.writer.add_scalar('VLG/current_reward', current_reward, self.trainer.iteration)
		self.writer.add_scalar('VLG/loss', loss_value, self.trainer.iteration)
		self.logger.write_to_log('grasp-success', self.trainer.grasp_success_log)

		# Save model snapshot
		if not self.is_testing:
			self.logger.save_backup_model(self.trainer.model, self.method)
			if self.trainer.iteration % 50 == 0:
				self.logger.save_model(self.trainer.iteration, self.trainer.model, self.method)
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