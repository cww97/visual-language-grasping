# !/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from skimage.segmentation import felzenszwalb
from torch.autograd import Variable


class Seg(nn.Module):
	'''
	segmentation
	get object candidate
	'''

	def forward(self, color_imgs, depth_imgs):
		'''
		get candidates for a batch
		return: candi
		'''
		batch_size = len(color_imgs)
		batch_imgs = []
		for color_img, depth_img in zip(color_imgs, depth_imgs):
			batch_imgs.append(self._get_candidate(color_img, depth_img))

		max_k = -1
		for img in batch_imgs:
			max_k = max(max_k, img.shape[0])
			
		is_valid = torch.zeros(batch_size, max_k)
		candidates = torch.zeros(tuple([batch_size, max_k]) + tuple(img.shape[1:]))
		for i, img in enumerate(batch_imgs):
			is_valid[i, 0: img.shape[0]] = 1
			candidates[i, 0: img.shape[0]] = img

		return (candidates.cuda(), is_valid.cuda())

	def _get_candidate(self, color_img, depth_img):
		denoise_img = self._remove_noise(color_img)
		segments_fz = felzenszwalb(denoise_img, scale=28000, min_size=150)
		num_seg = len(np.unique(segments_fz))

		depth_img = depth_img[:, :, np.newaxis]
		color_depth_img = np.concatenate((color_img, depth_img), axis=2)
		obj_imgs = []
		for i in range(1, num_seg):
			mask = np.zeros_like(color_depth_img)
			mask[segments_fz == i] = True
			obj_imgs.append(color_depth_img * mask)
			# import cv2
			# cv2.imwrite('test1.png', color_img)
			# cv2.imwrite('test.png', obj_imgs[-1][:, :, 0: 3])

		obj_imgs = torch.Tensor(obj_imgs).permute(0, 3, 1, 2)
		return obj_imgs

	def _remove_noise(self, img):
		n, m, _ = img.shape
		img = ndimage.median_filter(img, 2)
		# _remove_shadow()
		for i in range(n):
			for j in range(m):
				if np.sum(img[i][j] == 0) == 3 or ((img[i][j]>66).all() and (img[i][j]<82).all()):
					img[i][j] = np.array([45, 45, 45])
		return img
