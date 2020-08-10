import os
import time

import numpy as np
import yaml

import utils
from . import vrep
from ..robot import Robot as BaseRobot
from ..robot import Reward
from ..data import Data as TextData
import random
from bisect import bisect_right


class SimRobot(BaseRobot):
	def __init__(self, obj_mesh_dir, num_obj, *args):
		BaseRobot.__init__(self, *args)
		self.text_data = TextData()

		# Define colors for object meshes (Tableau palette)
		self.color_space = np.asarray([[78.0, 121.0, 167.0],  # blue
										[89.0, 161.0, 79.0],  # green
										[156, 117, 95],  # brown
										[242, 142, 43],  # orange
										[237.0, 201.0, 72.0],  # yellow
										[186, 176, 172],  # gray
										[255.0, 87.0, 89.0],  # red
										[176, 122, 161],  # purple
										[118, 183, 178],  # cyan
										[255, 157, 167]]) / 255.0  # pink

		# Read files in object mesh directory
		self.obj_mesh_dir = obj_mesh_dir
		self.num_obj = num_obj
		self.mesh_list = list(filter(lambda x: x.endswith('.obj'), os.listdir(self.obj_mesh_dir)))

		try:
			with open(os.path.join(obj_mesh_dir, 'blocks.yml')) as f:
				yaml_dict = yaml.safe_load(f)
			self.groups = yaml_dict['groups']
			self.mesh_name = yaml_dict['names']
			for obj in self.mesh_list:
				if obj not in self.mesh_name.keys():
					raise Exception
		except Exception:
			print('Failed to read block names/groups')
			exit(1)

		# Make sure to have the server side running in V-REP:
		# in a child script of a V-REP scene, add following command
		# to be executed just once, at simulation start:
		#
		# simExtRemoteApiStart(19999)
		#
		# then start simulation, and run this program.
		#
		# IMPORTANT: for each successful call to simxStart, there
		# should be a corresponding call to simxFinish at the end!

		# MODIFY remoteApiConnections.txt

		# Connect to simulator
		vrep.simxFinish(-1)  # Just in case, close all opened connections
		# Connect to V-REP on port 19997
		self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
		if self.sim_client == -1:
			print('Failed to connect to simulation (V-REP remote API server). Exiting.')
			exit()
		else:
			print('Connected to simulation.')
			# self.restart_sim()
		self.MODE = vrep.simx_opmode_blocking 

		# Setup virtual camera in simulation
		self.setup_sim_camera()
		self.object_handles = []
		self.object_left_handles = []
		self.target_handle = None

		# Add objects to simulation environment
		# self.add_objects()

	def setup_sim_camera(self):

		# Get handle to camera
		sim_ret, self.cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_persp', vrep.simx_opmode_blocking)

		# Get camera pose and intrinsics in simulationo
		sim_ret, cam_position = vrep.simxGetObjectPosition(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
		sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.cam_handle, -1, vrep.simx_opmode_blocking)
		cam_trans = np.eye(4, 4)
		cam_trans[0:3, 3] = np.asarray(cam_position)
		cam_orientation = [-cam_orientation[0], -cam_orientation[1], -cam_orientation[2]]
		cam_rotm = np.eye(4, 4)
		cam_rotm[0:3, 0:3] = np.linalg.inv(utils.euler2rotm(cam_orientation))
		# Compute rigid transformation representating camera pose
		self.cam_pose = np.dot(cam_trans, cam_rotm)
		self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
		self.cam_depth_scale = 1

		# Get background image
		self.bg_color_img, self.bg_depth_img = self.get_camera_data()
		self.bg_depth_img = self.bg_depth_img * self.cam_depth_scale

	def add_objects(self):
		# Randomly choose objects to add to scene
		group_chosen = np.random.choice(self.groups, size=self.num_obj, replace=False)
		self.obj_mesh_ind = np.array([self.mesh_list.index(np.random.choice(obj)) for obj in group_chosen])
		# TODO
		# handle <-> ind <-> obj -> name
		# Just for debug
		# print([self.mesh_list[ind] for ind in self.obj_mesh_ind])
		# self.obj_mesh_ind = np.array(range(len(self.mesh_list)))

		# self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)) % 10, :]
		self.obj_mesh_color = self.color_space[np.random.choice(np.arange(self.color_space.shape[0]), size=self.num_obj, replace=False)]
		# Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
		self.object_handles = []
		for object_idx in range(len(self.obj_mesh_ind)):
			curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])
			curr_shape_name = 'shape_%02d' % object_idx
			drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
			drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
			object_position = [drop_x, drop_y, 0.15]
			object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample()]
			object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1], self.obj_mesh_color[object_idx][2]]
			ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer', vrep.sim_scripttype_childscript, 'importShape', [0, 0, 255, 0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
			if ret_resp == 8:
				print('Failed to add new objects to simulation. Please restart.')
				exit()
			# print(ret_ints, ret_ints[0])
			curr_shape_handle = ret_ints[0]
			self.object_handles.append(curr_shape_handle)
			time.sleep(2)
		self.object_left_handles = self.object_handles.copy()
		self.prev_obj_positions = []
		self.obj_positions = []
		self.get_instruction()  # nb

	def restart_sim(self):
		sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_target', vrep.simx_opmode_blocking)
		vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5, 0, 0.3), vrep.simx_opmode_blocking)
		vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
		vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
		time.sleep(1)
		sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
		sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
		# V-REP bug requiring multiple starts and stops to restart
		while gripper_position[2] > 0.4:
			vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
			vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
			time.sleep(1)
			sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)

	def is_stable(self):
		# Check if simulation is stable by checking if gripper is within workspace
		sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
		sim_is_ok = gripper_position[0] > self.workspace_limits[0][0] - 0.1 and \
					gripper_position[0] < self.workspace_limits[0][1] + 0.1 and \
					gripper_position[1] > self.workspace_limits[1][0] - 0.1 and \
					gripper_position[1] < self.workspace_limits[1][1] + 0.1 and \
					gripper_position[2] > self.workspace_limits[2][0] and \
					gripper_position[2] < self.workspace_limits[2][1]
		if not sim_is_ok:
			print('Simulation unstable, Reset.')
		return sim_is_ok

	def reset(self):
		self.restart_sim()
		self.add_objects()

	# def stop_sim(self):objects/blocks
	#     if self.is_sim:
	#         # Now send some data to V-REP in a non-blocking fashion:
	#         # vrep.simxAddStatusbarMessage(sim_client,'Hello V-REP!',vrep.simx_opmode_oneshot)

	#         # # Start the simulation
	#         # vrep.simxStartSimulation(sim_client,vrep.simx_opmode_oneshot_wait)

	#         # # Stop simulation:
	#         # vrep.simxStopSimulation(sim_client,vrep.simx_opmode_oneshot_wait)

	#         # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
	#         vrep.simxGetPingTime(self.sim_client)

	#         # Now close the connection to V-REP:
	#         vrep.simxFinish(self.sim_client)

	def get_task_score(self):

		key_positions = np.asarray([[-0.625, 0.125, 0.0],  # red
									[-0.625, -0.125, 0.0],  # blue
									[-0.375, 0.125, 0.0],  # green
									[-0.375, -0.125, 0.0]])  # yellow

		obj_positions = np.asarray(self.get_obj_positions())
		obj_positions.shape = (1, obj_positions.shape[0], obj_positions.shape[1])
		obj_positions = np.tile(obj_positions, (key_positions.shape[0], 1, 1))

		key_positions.shape = (key_positions.shape[0], 1, key_positions.shape[1])
		key_positions = np.tile(key_positions, (1, obj_positions.shape[1], 1))

		key_dist = np.sqrt(np.sum(np.power(obj_positions - key_positions, 2), axis=2))
		key_nn_idx = np.argmin(key_dist, axis=0)

		return np.sum(key_nn_idx == np.asarray(range(self.num_obj)) % 4)

	def check_goal_reached(self, handle):
		# goal_reached = self.get_task_score() == self.num_obj
		goal_reached = self.target_handle == handle
		return goal_reached

	def get_obj_positions(self):

		obj_positions = []
		for object_handle in self.object_handles:
			sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
			obj_positions.append(object_position)

		return obj_positions

	def get_obj_positions_and_orientations(self):

		obj_positions = []
		obj_orientations = []
		for object_handle in self.object_handles:
			sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
			sim_ret, object_orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
			obj_positions.append(object_position)
			obj_orientations.append(object_orientation)

		return obj_positions, obj_orientations

	def reposition_objects(self, workspace_limits):
		# Move gripper out of the way
		self.move_to([-0.1, 0, 0.3], None)
		# sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
		# vrep.simxSetObjectPosition(self.sim_client, UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
		# time.sleep(1)

		for object_handle in self.object_handles:
			# Drop object at random x,y location and random orientation in robot workspace
			drop_x = (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample() + workspace_limits[0][0] + 0.1
			drop_y = (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample() + workspace_limits[1][0] + 0.1
			object_position = [drop_x, drop_y, 0.15]
			object_orientation = [2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample(), 2 * np.pi * np.random.random_sample()]
			vrep.simxSetObjectPosition(self.sim_client, object_handle, -1, object_position, vrep.simx_opmode_blocking)
			vrep.simxSetObjectOrientation(self.sim_client, object_handle, -1, object_orientation, vrep.simx_opmode_blocking)
			time.sleep(2)

	def get_camera_data(self):
		# Get color image from simulation
		sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, self.cam_handle, 0, vrep.simx_opmode_blocking)
		color_img = np.asarray(raw_image)
		color_img.shape = (resolution[1], resolution[0], 3)
		color_img = color_img.astype(np.float) / 255
		color_img[color_img < 0] += 1
		color_img *= 255
		color_img = np.fliplr(color_img)
		color_img = color_img.astype(np.uint8)

		# Get depth image from simulation
		sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, self.cam_handle, vrep.simx_opmode_blocking)
		depth_img = np.asarray(depth_buffer)
		depth_img.shape = (resolution[1], resolution[0])
		depth_img = np.fliplr(depth_img)
		zNear = 0.01
		zFar = 10
		depth_img = depth_img * (zFar - zNear) + zNear

		return color_img, depth_img

	def get_instruction(self):
		# TODO
		# add more template
		instruction_template = "pick up the {color} {shape}."
		ind = np.random.randint(0, self.num_obj)
		color = utils.get_mush_color_name(self.obj_mesh_color[ind])
		shape = np.random.choice(self.mesh_name[self.mesh_list[self.obj_mesh_ind[ind]]])
		self.target_handle = self.object_handles[ind]
		self.instruction_str = instruction_template.format(color=color, shape=shape)  # nb
		self.instruction = self.text_data.get_tensor(self.instruction_str)
		return self.instruction

	def close_gripper(self, _async=False):
		gripper_motor_velocity = -0.5
		gripper_motor_force = 100
		sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
		sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
		vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
		vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
		gripper_fully_closed = False
		while gripper_joint_position > -0.047:  # Block until gripper is fully closed
			sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
			# print(gripper_joint_position)
			if new_gripper_joint_position >= gripper_joint_position:
				return gripper_fully_closed
			gripper_joint_position = new_gripper_joint_position
		gripper_fully_closed = True

		return gripper_fully_closed

	def open_gripper(self, _async=False):
		gripper_motor_velocity = 0.5
		gripper_motor_force = 20
		sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
		sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
		vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
		vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
		while gripper_joint_position < 0.0536:  # Block until gripper is fully open
			sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)

	def move_to(self, tool_position, tool_orientation):
		# sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
		sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)

		move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
		move_magnitude = np.linalg.norm(move_direction)
		move_step = 0.02 * move_direction / move_magnitude
		num_move_steps = int(np.floor(move_magnitude / 0.02))

		for step_iter in range(num_move_steps):
			vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]), vrep.simx_opmode_blocking)
			sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
		vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)

	# Primitives ----------------------------------------------------------

	def random_grasp_action(self):
		'''
		angles = []
		for i in range(8):
			angle = np.deg2rad(i * (360.0 / 16))
			tool_rotation_angle = (angle % np.pi) - np.pi / 2
			angles.append(tool_rotation_angle)
		print(angles)
		'''
		# assert len(self.object_left_handles) > 0
		object_handle = random.sample(self.object_left_handles, 1)[0]
		
		_, orientation = vrep.simxGetObjectOrientation(self.sim_client, object_handle, -1, self.MODE)
		all_angles = [-1.5708, -1.1781, -0.7854, -0.3927, 0.0, 0.3927, 0.7854, 1.1781]
		possible_angles = [orientation[1], orientation[1] - np.pi/2.0]
		anegle = random.sample(possible_angles, 1)[0]
		angle = max(0, bisect_right(all_angles, orientation[1]) - 1)
		
		_, position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, self.MODE)
		action_x = (position[1] - self.workspace_limits[1][0]) / self.heightmap_resolution
		action_y = (position[0] - self.workspace_limits[0][0]) / self.heightmap_resolution
		action_x = min(action_x, 223)
		action_y = min(action_y, 223)
		action = (angle, int(action_x), int(action_y))
		# print(object_handle, action)
		# import pdb; pdb.set_trace()
		return action
	
	def step(self, action, valid_depth_heightmap, num_rotations, heightmap_resolution):
		# Compute 3D position of pixel
		angle = np.deg2rad(action[0] * (360.0 / num_rotations))
		best_pix_x = action[2]
		best_pix_y = action[1]
		primitive_position = [
			best_pix_x * heightmap_resolution + self.workspace_limits[0][0], 
			best_pix_y * heightmap_resolution + self.workspace_limits[1][0],
			valid_depth_heightmap[best_pix_y][best_pix_x] + self.workspace_limits[2][0]
		]

		reward = self.grasp(primitive_position, angle)
		done = (reward == Reward.SUCCESS)
		# print(reward, done)
		return reward.value, done

	def grasp(self, position, heightmap_rotation_angle):
		# print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))
		# Compute tool orientation from heightmap rotation angle
		tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2

		# Avoid collision with floor
		position = np.asarray(position).copy()
		position[2] = max(position[2] - 0.04, self.workspace_limits[2][0] + 0.02)

		# Move gripper to location above grasp target
		grasp_location_margin = 0.15
		# sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
		location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)

		# Compute gripper position and linear movement increments
		tool_position = location_above_grasp_target
		sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
		move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
		move_magnitude = np.linalg.norm(move_direction)
		move_step = 0.05 * move_direction / move_magnitude
		# if np.floor(move_direction[0] / move_step[0]) == np.nan or move_step[0] == 0: import pdb; pdb.set_trace() 
		num_move_steps = int(np.floor(move_direction[0] / move_step[0])) if move_step[0] != 0 else 1

		# Compute gripper orientation and rotation increments
		sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
		rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
		num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

		# Simultaneously move and rotate gripper
		for step_iter in range(max(num_move_steps, num_rotation_steps)):
			vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps), UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps), UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
			vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2), vrep.simx_opmode_blocking)
		vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
		vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)

		# Ensure gripper is open
		self.open_gripper()

		# Approach grasp target
		self.move_to(position, None)

		# Close gripper to grasp target
		gripper_full_closed = self.close_gripper()

		# Move gripper to location above grasp target
		self.move_to(location_above_grasp_target, None)

		# Check if grasp is successful
		gripper_full_closed = self.close_gripper()
		grasp_sth = not gripper_full_closed

		# Move the grasped object elsewhere
		if grasp_sth:
			object_positions = np.asarray(self.get_obj_positions())
			object_positions = object_positions[:, 2]
			grasped_object_ind = np.argmax(object_positions)
			grasped_object_handle = self.object_handles[grasped_object_ind]
			vrep.simxSetObjectPosition(self.sim_client, grasped_object_handle, -1, (-0.5, 0.5 + 0.05 * float(grasped_object_ind), 0.1), self.MODE)
			self.object_left_handles.remove(grasped_object_handle)
			if grasped_object_handle == self.target_handle:
				return Reward.SUCCESS
			else:
				return Reward.WRONG
		else:
			return Reward.FAIL

	def push(self, position, heightmap_rotation_angle, workspace_limits):
		# print('Executing: push at (%f, %f, %f)' % (position[0], position[1], position[2]))
		# Compute tool orientation from heightmap rotation angle
		tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi / 2

		# Adjust pushing point to be on tip of finger
		position[2] = position[2] + 0.026

		# Compute pushing direction
		push_orientation = [1.0, 0.0]
		push_direction = np.asarray([push_orientation[0] * np.cos(heightmap_rotation_angle) - push_orientation[1] * np.sin(heightmap_rotation_angle), push_orientation[0] * np.sin(heightmap_rotation_angle) + push_orientation[1] * np.cos(heightmap_rotation_angle)])

		# Move gripper to location above pushing point
		pushing_point_margin = 0.1
		location_above_pushing_point = (position[0], position[1], position[2] + pushing_point_margin)

		# Compute gripper position and linear movement increments
		tool_position = location_above_pushing_point
		sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
		move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
		move_magnitude = np.linalg.norm(move_direction)
		move_step = 0.05 * move_direction / move_magnitude
		num_move_steps = int(np.floor(move_direction[0] / move_step[0]))

		# Compute gripper orientation and rotation increments
		sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
		rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
		num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1]) / rotation_step))

		# Simultaneously move and rotate gripper
		for step_iter in range(max(num_move_steps, num_rotation_steps)):
			vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (UR5_target_position[0] + move_step[0] * min(step_iter, num_move_steps), UR5_target_position[1] + move_step[1] * min(step_iter, num_move_steps), UR5_target_position[2] + move_step[2] * min(step_iter, num_move_steps)), vrep.simx_opmode_blocking)
			vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi / 2, gripper_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi / 2), vrep.simx_opmode_blocking)
		vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (tool_position[0], tool_position[1], tool_position[2]), vrep.simx_opmode_blocking)
		vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi / 2, tool_rotation_angle, np.pi / 2), vrep.simx_opmode_blocking)

		# Ensure gripper is closed
		self.close_gripper()

		# Approach pushing point
		self.move_to(position, None)

		# Compute target location (push to the right)
		push_length = 0.1
		target_x = min(max(position[0] + push_direction[0] * push_length, workspace_limits[0][0]), workspace_limits[0][1])
		target_y = min(max(position[1] + push_direction[1] * push_length, workspace_limits[1][0]), workspace_limits[1][1])
		push_length = np.sqrt(np.power(target_x - position[0], 2) + np.power(target_y - position[1], 2))

		# Move in pushing direction towards target location
		self.move_to([target_x, target_y, position[2]], None)

		# Move gripper to location above grasp target
		self.move_to([target_x, target_y, location_above_pushing_point[2]], None)

		push_success = True
		return push_success

	# def place(self, position, heightmap_rotation_angle, workspace_limits):
	#     print('Executing: place at (%f, %f, %f)' % (position[0], position[1], position[2]))

	#    # Compute tool orientation from heightmap rotation angle
	#    tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

	#    # Avoid collision with floor
	#    position[2] = max(position[2] + 0.04 + 0.02, workspace_limits[2][0] + 0.02)

	#    # Move gripper to location above place target
	#    place_location_margin = 0.1
	#    sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
	#    location_above_place_target = (position[0], position[1], position[2] + place_location_margin)
	#    self.move_to(location_above_place_target, None)

	#    sim_ret,gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, UR5_target_handle, -1, vrep.simx_opmode_blocking)
	#    if tool_rotation_angle - gripper_orientation[1] > 0:
	#        increment = 0.2
	#    else:
	#        increment = -0.2
	#    while abs(tool_rotation_angle - gripper_orientation[1]) >= 0.2:
	#        vrep.simxSetObjectOrientation(self.sim_client, UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + increment, np.pi/2), vrep.simx_opmode_blocking)
	#        time.sleep(0.01)
	#        sim_ret,gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, UR5_target_handle, -1, vrep.simx_opmode_blocking)
	#    vrep.simxSetObjectOrientation(self.sim_client, UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

	#    # Approach place target
	#    self.move_to(position, None)

	#    # Ensure gripper is open
	#    self.open_gripper()

	#    # Move gripper to location above place target
	#    self.move_to(location_above_place_target, None)
