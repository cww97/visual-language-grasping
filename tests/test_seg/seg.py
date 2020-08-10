import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from scipy import ndimage

def remove_noise(img):
	n, m, _ = img.shape
	img = ndimage.median_filter(img, 2)
	# _remove_shadow()
	for i in range(n):
		for j in range(m):
			if np.sum(img[i][j] == 0) == 3 or ((img[i][j]>66).all() and (img[i][j]<82).all()):
				img[i][j] = np.array([45, 45, 45])
	return img

origianl_img = cv2.imread('000000.0.color.image.png')
origianl_img = cv2.resize(origianl_img, (224, 224))
rotate_img = cv2.imread('000000.0.color.png')
img = remove_noise(rotate_img)

# Felzenszwalbs's method
segments_fz = felzenszwalb(img, scale=10000, min_size=150)
# segments_fz = felzenszwalb(img, scale=5, sigma=0.88, min_size=400)
with open('text.txt', 'w') as f:
	for i in range(224):
		for j in range(224):
			f.write(str(segments_fz[i][j]))
		f.write('\n')
print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")

fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
ax[0][0].imshow(origianl_img)
ax[0][0].set_title('Origianl Image')
ax[0][1].imshow(rotate_img)
ax[0][1].set_title('Rotate in 3D')
ax[1][0].imshow(img)
ax[1][0].set_title('DeNoise')
ax[1][1].imshow(mark_boundaries(img, segments_fz))
ax[1][1].set_title("Segmention")
for a in ax.ravel():
    a.set_axis_off()
plt.tight_layout()
plt.savefig("filename.png")
