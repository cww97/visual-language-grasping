import os
from envs.simulation.robot import SimRobot
from models.stage1 import Seg
from models import YoungNet, State, SelectNet
import utils
import cv2
import numpy as np
import torch
from envs.data import Data as TextData
import random
from itertools import count
from envs.data import Instruction
import torch.nn.functional as F
from logger import Logger
from tensorboardX import SummaryWriter
from collections import namedtuple
Stage2Data = namedtuple('Stage2Data', ('instruction', 'candidates', 'label'))


class ReplayMemory(object):

	def __init__(self, capacity=1024):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, transition):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = transition
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


class Trainer():

	def __init__(self, robot_args, logger_args):
		robot_args[1] = 1  # num_objs
		self.robot_args = robot_args
		self.data_path = os.getcwd() + '/logs/stage2data'

		# models
		self.obj_finder = Seg()
		# self.model = YoungNet(vocab_size=vocab, padding_idx=pad_idx).cuda()
		self.text_data = TextData()
		vocab, pad_idx = len(self.text_data.text_field.vocab), self.text_data.padding_idx

		# training
		self.obj_selector = SelectNet(vocab_size=vocab, padding_idx=pad_idx).cuda()
		self.optimizer = torch.optim.Adam(self.obj_selector.parameters())
		
		# logger
		self.logger = Logger(logger_args['continue_logging'], logger_args['logging_directory'], doing='TrainStage2')
		# self.logger.save_camera_info(self.robot.cam_intrinsics, self.robot.cam_pose, self.robot.cam_depth_scale)
		self.logger.save_heightmap_info(logger_args['workspace_limits'], logger_args['heightmap_resolution'])

		# $ tensorboard --host 0.0.0.0 --logdir runs
		self.writer = SummaryWriter()  # tensorboard

	def sample_stage2_data(self):
		path = self.data_path
		self.robot = SimRobot(*self.robot_args)

		for idx in range(8):
			for color in self.robot.color_space:
				for t in range(10):
					self.robot.restart_sim()
					self.robot.add_objects(mesh_idx=idx, mesh_color=color)
					color_name = utils.get_mush_color_name(color)
					this_path = '{}/{}_{}/{}/'.format(path, idx, color_name, t)
					if not os.path.exists(this_path): os.makedirs(this_path)
					color0, depth0 = self.robot.get_camera_data(self.robot.up_cam_handle)
					color1, depth1 = self.robot.get_camera_data()

					cv2.imwrite(this_path + 'color0.png', color0)
					cv2.imwrite(this_path + 'depth0.png', depth0*255)
					cv2.imwrite(this_path + 'color1.png', color1)
					cv2.imwrite(this_path + 'depth1.png', depth1*255)
					with open(this_path + 'instruction.txt', 'w') as f:
						f.write(self.robot.instruction_str)
					print(idx, self.robot.instruction_str, t)
		
	def _read_a_img(self, name):
		img = cv2.imread(name)
		tensor = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2)
		return tensor

	def _pack_one_data(self, k=5):
		''' k <= 8
		'''
		files = os.listdir(self.data_path)
		files = random.sample(files, k)

		instructions, candidates = [], []
		for file in files:
			choosen = random.sample(os.listdir('{}/{}'.format(self.data_path, file)), 1)[0]
			this_path = '{}/{}/{}/'.format(self.data_path, file, choosen)

			instruction_str = open(this_path + 'instruction.txt').read()
			instruction = self.text_data.get_tensor(instruction_str)
			color0 = self._read_a_img(this_path + 'color0.png')
			depth0 = self._read_a_img(this_path + 'depth0.png')
			# color1 = self._read_a_img(this_path + 'color1.png')
			# depth1 = self._read_a_img(this_path + 'depth1.png')
			img = torch.cat((color0, depth0), dim=1)

			instructions.append(instruction)
			candidates.append(img)
			
		ans = random.randint(0, k-1)
		# print(k, ans, len(instructions))
		# import pdb; pdb.set_trace()
		one_data = Stage2Data(instructions[ans], torch.cat(candidates), ans)
		return one_data

	def training(self, training_interval=128):
		self.memory = ReplayMemory(1024)
		cnt = 0
		for t in count():
			k = random.randint(2, 10)
			one_data = self._pack_one_data(k)
			self.memory.push(one_data)
			if (t+1) % training_interval == 0:
				loss, accruacy = self.optimize_model()

				self.writer.add_scalar('loss', loss, cnt)
				self.writer.add_scalar('accruay', accruacy, cnt)
				print('epoch = {}, loss = {}, accruacy = {}'.format(cnt, loss, accruacy))

				# Save model snapshot
				self.logger.save_backup_model(self.obj_selector, 'reinforcement')
				if cnt % 50 == 0:
					self.logger.save_model(cnt, self.obj_selector, 'reinforcement')
					self.obj_selector = self.obj_selector.cuda()
												
				cnt += 1
				if cnt > 100000: break
		
	def optimize_model(self, Batch_Size=128):
		batch_size = 8
		repeat_time = Batch_Size // batch_size
		loss_value, accruacy = 0.0, 0.0

		for t in range(repeat_time):
			transitions = self.memory.sample(batch_size)
			if transitions == None: return None
			batch = Stage2Data(*zip(*transitions))
			instr_batch = Instruction(*zip(*batch.instruction))
			image_batch = self.obj_finder.pack_data(batch_size, batch.candidates)
			label_batch = torch.tensor(batch.label).cuda()

			pred = self.obj_selector(instr_batch, image_batch)
			# import pdb; pdb.set_trace()
			_pred = pred.argmax(dim=1)
			accruacy += float((_pred == label_batch).sum()) / float((_pred.reshape(-1).size()[0]))

			loss = F.cross_entropy(pred, label_batch)
			loss_value += loss
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

		loss_value /= 1.0 * repeat_time
		accruacy /= 1.0 * repeat_time
		return loss_value, accruacy

	def test_accuracy(self):
		pass

	def main(self):
		# self.sample_stage2_data()
		# self.test_forward()
		self.training()
