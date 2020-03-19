from abc import ABCMeta
from abc import abstractmethod


class Robot(metaclass=ABCMeta):
    def __init__(self, workspace_limits):
        self.workspace_limits = workspace_limits

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