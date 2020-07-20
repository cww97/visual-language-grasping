from abc import ABCMeta
from abc import abstractmethod
from enum import Enum


class Reward(Enum):
    SUCCESS = 5.0
    FAIL = -2.0
    WRONG = -1.0


class Robot(metaclass=ABCMeta):
	def __init__(self, workspace_limits, heightmap_resolution):
		self.workspace_limits = workspace_limits
		self.heightmap_resolution = heightmap_resolution

	@abstractmethod
	def get_camera_data(self):
		pass

	@abstractmethod
	def close_gripper(self, _async=False):
		pass

	@abstractmethod
	def open_gripper(self, _async=False):
		pass

	@abstractmethod
	def move_to(self, tool_position, tool_orientation):
		pass

	@abstractmethod
	def grasp(self, position, heightmap_rotation_angle, workspace_limits):
		pass

	@abstractmethod
	def push(self, position, heightmap_rotation_angle, workspace_limits):
		pass
