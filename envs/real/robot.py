import socket
import struct
import time

import numpy as np

import utils

from .camera import Camera
from ..robot import Robot as BaseRobot

class RealRobot(BaseRobot):
    def __init__(self, tcp_host_ip, tcp_port, rtc_host_ip, rtc_port, workspace_limits):
        BaseRobot.__init__(self, workspace_limits)

        # Connect to robot client
        self.tcp_host_ip = tcp_host_ip
        self.tcp_port = tcp_port
        # self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Connect as real-time client to parse state data
        self.rtc_host_ip = rtc_host_ip
        self.rtc_port = rtc_port

        # Default home joint configuration
        # self.home_joint_config = [-np.pi, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
        self.home_joint_config = [-(180.0 / 360.0) * 2 * np.pi, -(84.2 / 360.0) * 2 * np.pi, (112.8 / 360.0) * 2 * np.pi, -(119.7 / 360.0) * 2 * np.pi, -(90.0 / 360.0) * 2 * np.pi, 0.0]

        # Default joint speed configuration
        self.joint_acc = 8  # Safe: 1.4
        self.joint_vel = 3  # Safe: 1.05

        # Joint tolerance for blocking calls
        self.joint_tolerance = 0.01

        # Default tool speed configuration
        self.tool_acc = 1.2  # Safe: 0.5
        self.tool_vel = 0.25  # Safe: 0.2

        # Tool pose tolerance for blocking calls
        self.tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]

        # Move robot to home pose
        self.close_gripper()
        self.go_home()

        # Fetch RGB-D data from RealSense camera
       
        self.camera = Camera()
        self.cam_intrinsics = self.camera.intrinsics

        # Load camera pose (from running calibrate.py), intrinsics and depth scale
        self.cam_pose = np.loadtxt('real/camera_pose.txt', delimiter=' ')
        self.cam_depth_scale = np.loadtxt('real/camera_depth_scale.txt', delimiter=' ')

    def get_camera_data(self):
        # Get color and depth image from ROS service
        color_img, depth_img = self.camera.get_data()
        # color_img = self.camera.color_data.copy()
        # depth_img = self.camera.depth_data.copy()
        return color_img, depth_img

    @staticmethod
    def parse_tcp_state_data(state_data, subpackage):

        # Read package header
        data_bytes = bytearray()
        data_bytes.extend(state_data)
        data_length = struct.unpack("!i", data_bytes[0:4])[0]
        robot_message_type = data_bytes[4]
        assert (robot_message_type == 16)
        byte_idx = 5

        # Parse sub-packages
        subpackage_types = {'joint_data': 1, 'cartesian_info': 4, 'force_mode_data': 7, 'tool_data': 2}
        while byte_idx < data_length:
            # package_length = int.from_bytes(data_bytes[byte_idx:(byte_idx+4)], byteorder='big', signed=False)
            package_length = struct.unpack("!i", data_bytes[byte_idx:(byte_idx + 4)])[0]
            byte_idx += 4
            package_idx = data_bytes[byte_idx]
            if package_idx == subpackage_types[subpackage]:
                byte_idx += 1
                break
            byte_idx += package_length - 4

        def parse_joint_data(data_bytes, byte_idx):
            actual_joint_positions = [0, 0, 0, 0, 0, 0]
            target_joint_positions = [0, 0, 0, 0, 0, 0]
            for joint_idx in range(6):
                actual_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
                target_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx + 8):(byte_idx + 16)])[0]
                byte_idx += 41
            return actual_joint_positions

        def parse_cartesian_info(data_bytes, byte_idx):
            actual_tool_pose = [0, 0, 0, 0, 0, 0]
            for pose_value_idx in range(6):
                actual_tool_pose[pose_value_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
                byte_idx += 8
            return actual_tool_pose

        def parse_tool_data(data_bytes, byte_idx):
            byte_idx += 2
            tool_analog_input2 = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
            return tool_analog_input2

        parse_functions = {'joint_data': parse_joint_data, 'cartesian_info': parse_cartesian_info, 'tool_data': parse_tool_data}
        return parse_functions[subpackage](data_bytes, byte_idx)

    def parse_rtc_state_data(self, state_data):
        # Read package header
        data_bytes = bytearray()
        data_bytes.extend(state_data)
        data_length = struct.unpack("!i", data_bytes[0:4])[0]
        assert (data_length == 812)
        byte_idx = 4 + 8 + 8 * 48 + 24 + 120
        TCP_forces = [0, 0, 0, 0, 0, 0]
        for joint_idx in range(6):
            TCP_forces[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx + 0):(byte_idx + 8)])[0]
            byte_idx += 8
        return TCP_forces

    def close_gripper(self, _async=False):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "set_digital_out(8,True)\n"
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()
        if _async:
            gripper_fully_closed = True
        else:
            time.sleep(1.5)
            gripper_fully_closed = self.check_grasp()
        return gripper_fully_closed

    def open_gripper(self, _async=False):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "set_digital_out(8,False)\n"
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()
        if not _async:
            time.sleep(1.5)

    def get_state(self):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        state_data = self.tcp_socket.recv(2048)
        self.tcp_socket.close()
        return state_data

    def move_to(self, tool_position, tool_orientation):
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (tool_position[0], tool_position[1], tool_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.tool_acc, self.tool_vel)
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches target tool position
        tcp_state_data = self.tcp_socket.recv(2048)
        actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
        while not all([np.abs(actual_tool_pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]
            # print([np.abs(actual_tool_pose[j] - tool_position[j]) for j in range(3)] + [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) for j in range(3,6)])
            tcp_state_data = self.tcp_socket.recv(2048)
            prev_actual_tool_pose = np.asarray(actual_tool_pose).copy()
            actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
            time.sleep(0.01)
        self.tcp_socket.close()

    def guarded_move_to(self, tool_position, tool_orientation):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.rtc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        self.rtc_socket.connect((self.rtc_host_ip, self.rtc_port))

        # Read actual tool position
        tcp_state_data = self.tcp_socket.recv(2048)
        actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
        execute_success = True

        # Increment every cm, check force
        self.tool_acc = 0.1  # 1.2 # 0.5

        while not all([np.abs(actual_tool_pose[j] - tool_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            # [min(np.abs(actual_tool_pose[j] - tool_orientation[j-3]), np.abs(np.abs(actual_tool_pose[j] - tool_orientation[j-3]) - np.pi*2)) < self.tool_pose_tolerance[j] for j in range(3,6)]

            # Compute motion trajectory in 1cm increments
            increment = np.asarray([(tool_position[j] - actual_tool_pose[j]) for j in range(3)])
            if np.linalg.norm(increment) < 0.01:
                increment_position = tool_position
            else:
                increment = 0.01 * increment / np.linalg.norm(increment)
                increment_position = np.asarray(actual_tool_pose[0:3]) + increment

            # Move to next increment position (blocking call)
            tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (increment_position[0], increment_position[1], increment_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.tool_acc, self.tool_vel)
            self.tcp_socket.send(str.encode(tcp_command))

            time_start = time.time()
            tcp_state_data = self.tcp_socket.recv(2048)
            actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
            while not all([np.abs(actual_tool_pose[j] - increment_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                # print([np.abs(actual_tool_pose[j] - increment_position[j]) for j in range(3)])
                tcp_state_data = self.tcp_socket.recv(2048)
                actual_tool_pose = self.parse_tcp_state_data(tcp_state_data, 'cartesian_info')
                time_snapshot = time.time()
                if time_snapshot - time_start > 1:
                    break
                time.sleep(0.01)

            # Reading TCP forces from real-time client connection
            rtc_state_data = self.rtc_socket.recv(6496)
            TCP_forces = self.parse_rtc_state_data(rtc_state_data)

            # If TCP forces in x/y exceed 20 Newtons, stop moving
            # print(TCP_forces[0:3])
            if np.linalg.norm(np.asarray(TCP_forces[0:2])) > 20 or (time_snapshot - time_start) > 1:
                print('Warning: contact detected! Movement halted. TCP forces: [%f, %f, %f]' % (TCP_forces[0], TCP_forces[1], TCP_forces[2]))
                execute_success = False
                break

            time.sleep(0.01)

        self.tool_acc = 1.2  # 1.2 # 0.5

        self.tcp_socket.close()
        self.rtc_socket.close()

        return execute_success

    def move_joints(self, joint_configuration):

        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "movej([%f" % joint_configuration[0]
        for joint_idx in range(1, 6):
            tcp_command = tcp_command + (",%f" % joint_configuration[joint_idx])
        tcp_command = tcp_command + "],a=%f,v=%f)\n" % (self.joint_acc, self.joint_vel)
        self.tcp_socket.send(str.encode(tcp_command))

        # Block until robot reaches home state
        state_data = self.tcp_socket.recv(2048)
        actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
        while not all([np.abs(actual_joint_positions[j] - joint_configuration[j]) < self.joint_tolerance for j in range(6)]):
            state_data = self.tcp_socket.recv(2048)
            actual_joint_positions = self.parse_tcp_state_data(state_data, 'joint_data')
            time.sleep(0.01)

        self.tcp_socket.close()

    def go_home(self):
        self.move_joints(self.home_joint_config)

    # Note: must be preceded by close_gripper()
    def check_grasp(self):

        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        return tool_analog_input2 > 0.26

    # Primitives ----------------------------------------------------------

    def grasp(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))

        # Compute tool orientation from heightmap rotation angle
        grasp_orientation = [1.0, 0.0]
        if heightmap_rotation_angle > np.pi:
            heightmap_rotation_angle = heightmap_rotation_angle - 2 * np.pi
        tool_rotation_angle = heightmap_rotation_angle / 2
        tool_orientation = np.asarray([grasp_orientation[0] * np.cos(tool_rotation_angle) - grasp_orientation[1] * np.sin(tool_rotation_angle), grasp_orientation[0] * np.sin(tool_rotation_angle) + grasp_orientation[1] * np.cos(tool_rotation_angle), 0.0]) * np.pi
        tool_orientation_angle = np.linalg.norm(tool_orientation)
        tool_orientation_axis = tool_orientation / tool_orientation_angle
        tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3, :3]

        # Compute tilted tool orientation during dropping into bin
        tilt_rotm = utils.euler2rotm(np.asarray([-np.pi / 4, 0, 0]))
        tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
        tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
        tilted_tool_orientation = tilted_tool_orientation_axis_angle[0] * np.asarray(tilted_tool_orientation_axis_angle[1:4])

        # Attempt grasp
        position = np.asarray(position).copy()
        position[2] = max(position[2] - 0.05, workspace_limits[2][0])
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "def process():\n"
        tcp_command += " set_digital_out(8,False)\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (position[0], position[1], position[2] + 0.1, tool_orientation[0], tool_orientation[1], 0.0, self.joint_acc * 0.5, self.joint_vel * 0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (position[0], position[1], position[2], tool_orientation[0], tool_orientation[1], 0.0, self.joint_acc * 0.1, self.joint_vel * 0.1)
        tcp_command += " set_digital_out(8,True)\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()

        # Block until robot reaches target tool position and gripper fingers have stopped moving
        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        timeout_t0 = time.time()
        while True:
            state_data = self.get_state()
            new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            timeout_t1 = time.time()
            if (tool_analog_input2 < 3.7 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - position[j]) < self.tool_pose_tolerance[j] for j in range(3)])) or (timeout_t1 - timeout_t0) > 5:
                break
            tool_analog_input2 = new_tool_analog_input2

        # Check if gripper is open (grasp might be successful)
        gripper_open = tool_analog_input2 > 0.26

        # # Check if grasp is successful
        # grasp_success =  tool_analog_input2 > 0.26

        home_position = [0.49, 0.11, 0.03]
        bin_position = [0.5, -0.45, 0.1]

        # If gripper is open, drop object in bin and check if grasp is successful
        grasp_success = False
        if gripper_open:

            # Pre-compute blend radius
            blend_radius = min(abs(bin_position[1] - position[1]) / 2 - 0.01, 0.2)

            # Attempt placing
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "def process():\n"
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % (position[0], position[1], bin_position[2], tool_orientation[0], tool_orientation[1], 0.0, self.joint_acc, self.joint_vel, blend_radius)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=%f)\n" % (bin_position[0], bin_position[1], bin_position[2], tilted_tool_orientation[0], tilted_tool_orientation[1], tilted_tool_orientation[2], self.joint_acc, self.joint_vel, blend_radius)
            tcp_command += " set_digital_out(8,False)\n"
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.0)\n" % (home_position[0], home_position[1], home_position[2], tool_orientation[0], tool_orientation[1], 0.0, self.joint_acc * 0.5, self.joint_vel * 0.5)
            tcp_command += "end\n"
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()
            # print(tcp_command) # Debug

            # Measure gripper width until robot reaches near bin location
            state_data = self.get_state()
            measurements = []
            while True:
                state_data = self.get_state()
                tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
                actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
                measurements.append(tool_analog_input2)
                if abs(actual_tool_pose[1] - bin_position[1]) < 0.2 or all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                    break

            # If gripper width did not change before reaching bin location, then object is in grip and grasp is successful
            if len(measurements) >= 2:
                if abs(measurements[0] - measurements[1]) < 0.1:
                    grasp_success = True

        else:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
            tcp_command = "def process():\n"
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (position[0], position[1], position[2] + 0.1, tool_orientation[0], tool_orientation[1], 0.0, self.joint_acc * 0.5, self.joint_vel * 0.5)
            tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.0)\n" % (home_position[0], home_position[1], home_position[2], tool_orientation[0], tool_orientation[1], 0.0, self.joint_acc * 0.5, self.joint_vel * 0.5)
            tcp_command += "end\n"
            self.tcp_socket.send(str.encode(tcp_command))
            self.tcp_socket.close()

        # Block until robot reaches home location
        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
        while True:
            state_data = self.get_state()
            new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            if (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
            tool_analog_input2 = new_tool_analog_input2

        return grasp_success

    def push(self, position, heightmap_rotation_angle, workspace_limits):
        print('Executing: push at (%f, %f, %f)' % (position[0], position[1], position[2]))

        # Compute tool orientation from heightmap rotation angle
        push_orientation = [1.0, 0.0]
        tool_rotation_angle = heightmap_rotation_angle / 2
        tool_orientation = np.asarray([push_orientation[0] * np.cos(tool_rotation_angle) - push_orientation[1] * np.sin(tool_rotation_angle), push_orientation[0] * np.sin(tool_rotation_angle) + push_orientation[1] * np.cos(tool_rotation_angle), 0.0]) * np.pi
        tool_orientation_angle = np.linalg.norm(tool_orientation)
        tool_orientation_axis = tool_orientation / tool_orientation_angle
        tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3, :3]

        # Compute push direction and endpoint (push to right of rotated heightmap)
        push_direction = np.asarray([push_orientation[0] * np.cos(heightmap_rotation_angle) - push_orientation[1] * np.sin(heightmap_rotation_angle), push_orientation[0] * np.sin(heightmap_rotation_angle) + push_orientation[1] * np.cos(heightmap_rotation_angle), 0.0])
        target_x = min(max(position[0] + push_direction[0] * 0.1, workspace_limits[0][0]), workspace_limits[0][1])
        target_y = min(max(position[1] + push_direction[1] * 0.1, workspace_limits[1][0]), workspace_limits[1][1])
        push_endpoint = np.asarray([target_x, target_y, position[2]])
        push_direction.shape = (3, 1)

        # Compute tilted tool orientation during push
        tilt_axis = np.dot(utils.euler2rotm(np.asarray([0, 0, np.pi / 2]))[:3, :3], push_direction)
        tilt_rotm = utils.angle2rotm(-np.pi / 8, tilt_axis, point=None)[:3, :3]
        tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
        tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
        tilted_tool_orientation = tilted_tool_orientation_axis_angle[0] * np.asarray(tilted_tool_orientation_axis_angle[1:4])

        # Push only within workspace limits
        position = np.asarray(position).copy()
        position[0] = min(max(position[0], workspace_limits[0][0]), workspace_limits[0][1])
        position[1] = min(max(position[1], workspace_limits[1][0]), workspace_limits[1][1])
        position[2] = max(position[2] + 0.005, workspace_limits[2][0] + 0.005)  # Add buffer to surface

        home_position = [0.49, 0.11, 0.03]

        # Attempt push
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "def process():\n"
        tcp_command += " set_digital_out(8,True)\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (position[0], position[1], position[2] + 0.1, tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc * 0.5, self.joint_vel * 0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (position[0], position[1], position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc * 0.1, self.joint_vel * 0.1)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (push_endpoint[0], push_endpoint[1], push_endpoint[2], tilted_tool_orientation[0], tilted_tool_orientation[1], tilted_tool_orientation[2], self.joint_acc * 0.1, self.joint_vel * 0.1)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.03)\n" % (position[0], position[1], position[2] + 0.1, tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc * 0.5, self.joint_vel * 0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (home_position[0], home_position[1], home_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc * 0.5, self.joint_vel * 0.5)
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()

        # Block until robot reaches target tool position and gripper fingers have stopped moving
        state_data = self.get_state()
        while True:
            state_data = self.get_state()
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            if all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
        push_success = True
        time.sleep(0.5)

        return push_success

    def restart_real(self):
        # Compute tool orientation from heightmap rotation angle
        grasp_orientation = [1.0, 0.0]
        tool_rotation_angle = -np.pi / 4
        tool_orientation = np.asarray([grasp_orientation[0] * np.cos(tool_rotation_angle) - grasp_orientation[1] * np.sin(tool_rotation_angle), grasp_orientation[0] * np.sin(tool_rotation_angle) + grasp_orientation[1] * np.cos(tool_rotation_angle), 0.0]) * np.pi
        tool_orientation_angle = np.linalg.norm(tool_orientation)
        tool_orientation_axis = tool_orientation / tool_orientation_angle
        tool_orientation_rotm = utils.angle2rotm(tool_orientation_angle, tool_orientation_axis, point=None)[:3, :3]

        tilt_rotm = utils.euler2rotm(np.asarray([-np.pi / 4, 0, 0]))
        tilted_tool_orientation_rotm = np.dot(tilt_rotm, tool_orientation_rotm)
        tilted_tool_orientation_axis_angle = utils.rotm2angle(tilted_tool_orientation_rotm)
        tilted_tool_orientation = tilted_tool_orientation_axis_angle[0] * np.asarray(tilted_tool_orientation_axis_angle[1:4])

        # Move to box grabbing position
        box_grab_position = [0.5, -0.35, -0.12]
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "def process():\n"
        tcp_command += " set_digital_out(8,False)\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (box_grab_position[0], box_grab_position[1], box_grab_position[2] + 0.1, tilted_tool_orientation[0], tilted_tool_orientation[1], tilted_tool_orientation[2], self.joint_acc, self.joint_vel)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0], box_grab_position[1], box_grab_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc, self.joint_vel)
        tcp_command += " set_digital_out(8,True)\n"
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()

        # Block until robot reaches box grabbing position and gripper fingers have stopped moving
        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        while True:
            state_data = self.get_state()
            new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            if tool_analog_input2 < 3.7 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - box_grab_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
            tool_analog_input2 = new_tool_analog_input2

        # Move to box release position
        box_release_position = [0.5, 0.08, -0.12]
        home_position = [0.49, 0.11, 0.03]
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        tcp_command = "def process():\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_release_position[0], box_release_position[1], box_release_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc * 0.1, self.joint_vel * 0.1)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_release_position[0], box_release_position[1], box_release_position[2] + 0.3, tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc * 0.02, self.joint_vel * 0.02)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.29)\n" % (box_grab_position[0] - 0.05, box_grab_position[1] + 0.1, box_grab_position[2] + 0.3, tilted_tool_orientation[0], tilted_tool_orientation[1], tilted_tool_orientation[2], self.joint_acc * 0.5, self.joint_vel * 0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0] - 0.05, box_grab_position[1] + 0.1, box_grab_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc * 0.5, self.joint_vel * 0.5)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0], box_grab_position[1], box_grab_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc * 0.1, self.joint_vel * 0.1)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (box_grab_position[0] + 0.05, box_grab_position[1], box_grab_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc * 0.1, self.joint_vel * 0.1)
        tcp_command += " set_digital_out(8,False)\n"
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.09)\n" % (box_grab_position[0], box_grab_position[1], box_grab_position[2] + 0.1, tilted_tool_orientation[0], tilted_tool_orientation[1], tilted_tool_orientation[2], self.joint_acc, self.joint_vel)
        tcp_command += " movej(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0.00)\n" % (home_position[0], home_position[1], home_position[2], tool_orientation[0], tool_orientation[1], tool_orientation[2], self.joint_acc, self.joint_vel)
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))
        self.tcp_socket.close()

        # Block until robot reaches home position
        state_data = self.get_state()
        tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
        while True:
            state_data = self.get_state()
            new_tool_analog_input2 = self.parse_tcp_state_data(state_data, 'tool_data')
            actual_tool_pose = self.parse_tcp_state_data(state_data, 'cartesian_info')
            if tool_analog_input2 > 3.0 and (abs(new_tool_analog_input2 - tool_analog_input2) < 0.01) and all([np.abs(actual_tool_pose[j] - home_position[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
                break
            tool_analog_input2 = new_tool_analog_input2

    # def place(self, position, orientation, workspace_limits):
    #     print('Executing: place at (%f, %f, %f)' % (position[0], position[1], position[2]))

    #     # Attempt placing
    #     position[2] = max(position[2], workspace_limits[2][0])
    #     self.move_to([position[0], position[1], position[2] + 0.2], orientation)
    #     self.move_to([position[0], position[1], position[2] + 0.05], orientation)
    #     self.tool_acc = 1 # 0.05
    #     self.tool_vel = 0.02 # 0.02
    #     self.move_to([position[0], position[1], position[2]], orientation)
    #     self.open_gripper()
    #     self.tool_acc = 1 # 0.5
    #     self.tool_vel = 0.2 # 0.2
    #     self.move_to([position[0], position[1], position[2] + 0.2], orientation)
    #     self.close_gripper()
    #     self.go_home()
