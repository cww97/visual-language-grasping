import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.autograd import Variable
from models import reactive_net, reinforcement_net
from utils import CrossEntropyLoss2d
from envs.data import Data as TextData


class Trainer(object):

    def __init__(self, method, future_reward_discount,
                 is_testing, load_snapshot, snapshot_file):
        assert torch.cuda.is_available()
        self.method = method
        self.text_data = TextData()
        vocab, dim, drop = len(self.text_data.text_field.vocab), 32, 0.5

        # Fully convolutional classification network for supervised learning
        if self.method == 'reactive':
            self.model = reactive_net(embed_num=vocab, embed_dim=dim, drop_out=drop)

            # Initialize classification loss
            grasp_num_classes = 3  # 0 - grasp, 1 - failed grasp, 2 - no loss
            grasp_class_weights = torch.ones(grasp_num_classes)
            grasp_class_weights[grasp_num_classes - 1] = 0
            self.grasp_criterion = CrossEntropyLoss2d(grasp_class_weights.cuda()).cuda()

        # Fully convolutional Q network for deep reinforcement learning
        elif self.method == 'reinforcement':
            self.model = reinforcement_net(embed_num=vocab, embed_dim=dim, drop_out=drop)
            self.future_reward_discount = future_reward_discount

            # Initialize Huber loss
            self.criterion = torch.nn.SmoothL1Loss(reduce=False).cuda()  # Huber losss

        # Load pre-trained model
        if load_snapshot:
            # PyTorch v0.4 removes periods in state dict keys, but no backwards compatibility :(
            loaded_snapshot_state_dict = torch.load(snapshot_file)
            items = ('conv.1', 'norm.1', 'conv.2', 'norm.2')
            for item in items:
                self.loaded_snapshot_item(loaded_snapshot_state_dict, item)
            self.model.load_state_dict(loaded_snapshot_state_dict)

            print('Pre-trained model snapshot loaded from: %s' % (snapshot_file))

        # Convert model from CPU to GPU
        self.model = self.model.cuda()

        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.iteration = 0

        # Initialize lists to save execution info and RL variables
        self.executed_action_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.predicted_value_log = []
        self.use_heuristic_log = []
        self.is_exploit_log = []
        self.clearance_log = []

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
        self.is_exploit_log = np.loadtxt(os.path.join(transitions_directory, 'is-exploit.log.txt'), delimiter=' ')
        self.is_exploit_log = self.is_exploit_log[0:self.iteration]
        self.is_exploit_log.shape = (self.iteration, 1)
        self.is_exploit_log = self.is_exploit_log.tolist()
        self.clearance_log = np.loadtxt(os.path.join(transitions_directory, 'clearance.log.txt'), delimiter=' ')
        self.clearance_log.shape = (self.clearance_log.shape[0], 1)
        self.clearance_log = self.clearance_log.tolist()

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

        # Pass input data through model, cslnb
        output_prob, state_feat = self.model.forward(
            instruction_tensor, input_color_data, input_depth_data, is_volatile, specific_rotation
        )
        # print(len(output_prob), type(output_prob[0]), output_prob[0])

        if self.method == 'reactive':

            # Return affordances (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    grasp_predictions = F.softmax(output_prob[rotate_idx][0], dim=1).cpu().data.numpy()[
                        :, 0, (padding_width // 2): (color_heightmap_2x.shape[0] // 2 - padding_width // 2),
                        (padding_width // 2): (color_heightmap_2x.shape[0] // 2 - padding_width // 2)
                    ]
                else:
                    grasp_predictions = np.concatenate((
                        grasp_predictions,
                        F.softmax(output_prob[rotate_idx][0], dim=1).cpu().data.numpy()[
                            :, 0, (padding_width // 2): (color_heightmap_2x.shape[0] // 2 - padding_width // 2),
                            (padding_width // 2): (color_heightmap_2x.shape[0] // 2 - padding_width // 2)
                        ]
                    ), axis=0)

        elif self.method == 'reinforcement':

            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    grasp_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[
                        :, 0, int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2),
                        int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2)]
                else:
                    grasp_predictions = np.concatenate((
                        grasp_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[
                            :, 0, int(padding_width / 2): int(color_heightmap_2x.shape[0] / 2 - padding_width / 2),
                            int(padding_width / 2): int(color_heightmap_2x.shape[0] / 2 - padding_width / 2)
                        ]
                    ), axis=0)

        # print("grasp_predictions = ", grasp_predictions)
        return grasp_predictions, state_feat

    def get_label_value(
        self, primitive_action,
        grasp_success,
        change_detected,
        prev_grasp_predictions,
        instruction,
        next_color_heightmap,
        next_depth_heightmap
    ):
        if self.method == 'reactive':
            # label: 0 - grasp, 1 - failed grasp, 2 - no loss
            label_value = 0 if grasp_success else 1
            print('Label value: %d' % (label_value))
            return label_value, label_value

        elif self.method == 'reinforcement':

            # Compute current reward, deal with put in the future
            current_reward = 1.0 if grasp_success else 0.0

            # Compute future reward
            if not change_detected and not grasp_success:
                future_reward = 0
            else:
                next_grasp_predictions, next_state_feat = self.forward(
                    instruction, next_color_heightmap, next_depth_heightmap, is_volatile=True
                )
                future_reward = np.max(next_grasp_predictions)

            print('Current reward: %f' % (current_reward))
            print('Future reward: %f' % (future_reward))
            expected_reward = current_reward + self.future_reward_discount * future_reward
            print('Expected reward: %f + %f x %f = %f' % (
                current_reward, self.future_reward_discount, future_reward, expected_reward
            ))
            return expected_reward, current_reward

    # Compute labels and back-propagate
    def backprop(self, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value):

        if self.method == 'reactive':

            # Compute fill value
            fill_value = 2

            # Compute labels
            label = np.zeros((1, 320, 320)) + fill_value
            action_area = np.zeros((224, 224))
            action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
            tmp_label = np.zeros((224, 224)) + fill_value
            tmp_label[action_area > 0] = label_value
            label[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label

            # Compute loss and backward pass
            self.optimizer.zero_grad()
            loss_value = 0

            if primitive_action == 'grasp':
                # Do for-ward pass with specified rotation (to save gradients)
                grasp_predictions, state_feat = self.forward(
                    color_heightmap, depth_heightmap,
                    is_volatile=False, specific_rotation=best_pix_ind[0]
                )

                loss = self.grasp_criterion(
                    self.model.output_prob[0][0],
                    Variable(torch.from_numpy(label).long().cuda())
                )
                loss.backward()
                try: loss_value += loss.cpu().data.numpy()[0]
                except IndexError: loss_value += loss.cpu().data.numpy()

                # Since grasping is symmetric, train with another for-ward pass of opposite rotation angle
                opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations / 2) % self.model.num_rotations

                grasp_predictions, state_feat = self.forward(
                    color_heightmap, depth_heightmap,
                    is_volatile=False, specific_rotation=opposite_rotate_idx
                )

                loss = self.grasp_criterion(
                    self.model.output_prob[0][0],
                    Variable(torch.from_numpy(label).long().cuda())
                )
                loss.backward()
                try: loss_value += loss.cpu().data.numpy()[0]
                except IndexError: loss_value += loss.cpu().data.numpy()

                loss_value = loss_value / 2

            print('Training loss: %f' % (loss_value))
            self.optimizer.step()

        elif self.method == 'reinforcement':

            # Compute labels
            label = np.zeros((1, 320, 320))
            action_area = np.zeros((224, 224))
            action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
            tmp_label = np.zeros((224, 224))
            tmp_label[action_area > 0] = label_value
            label[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label

            # Compute label mask
            label_weights = np.zeros(label.shape)
            tmp_label_weights = np.zeros((224, 224))
            tmp_label_weights[action_area > 0] = 1
            label_weights[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label_weights

            # Compute loss and backward pass
            self.optimizer.zero_grad()
            loss_value = 0

            if primitive_action == 'grasp':

                # Do for-ward pass with specified rotation (to save gradients)
                grasp_predictions, state_feat = self.forward(
                    color_heightmap, depth_heightmap,
                    is_volatile=False, specific_rotation=best_pix_ind[0]
                )

                loss = self.criterion(
                    self.model.output_prob[0][0].view(1, 320, 320),
                    Variable(torch.from_numpy(label).float().cuda())
                ) * Variable(torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
                loss = loss.sum()
                loss.backward()
                try:
                    loss_value = loss.cpu().data.numpy()[0]
                except IndexError:
                    loss_value = loss.cpu().data.numpy()

                opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations / 2) % self.model.num_rotations

                grasp_predictions, state_feat = self.forward(
                    color_heightmap, depth_heightmap,
                    is_volatile=False, specific_rotation=opposite_rotate_idx
                )

                loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320), Variable(
                    torch.from_numpy(label).float().cuda())) * Variable(
                        torch.from_numpy(label_weights).float().cuda(), requires_grad=False)

                loss = loss.sum()
                loss.backward()
                try: loss_value = loss.cpu().data.numpy()[0]
                except IndexError: loss_value = loss.cpu().data.numpy()

                loss_value = loss_value / 2

            print('Training loss: %f' % (loss_value))
            self.optimizer.step()

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
    text_data = TextData()
    t = text_data.get_tensor('balabala dsa pick up, the')
    print(t)
