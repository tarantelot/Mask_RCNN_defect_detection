import os
import numpy as np
import skimage.data
from skimage.io import imsave, imread
from skimage import transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


img_w = 128
img_h = 128

PATH_NEW_IMGS_FOLDER = 'Resized_images'


def load_imgs():
	CONDITIONS = lambda img_name: False if '_mask' in img_name or '.json' in img_name or '.py' in img_name else True

	img_names = [img_name for img_name in os.listdir() if CONDITIONS(img_name) and os.path.isfile(img_name)]
	imgs = [imread(img_name) for img_name in img_names]

	return imgs, img_names


def resize(imgs):
	resized_imgs = [transform.resize(img, (img_w, img_h)) for img in imgs]
	return resized_imgs


def save_imgs(imgs, img_names):
	for img, img_name in zip(imgs, img_names):
		imsave(os.path.join(PATH_NEW_IMGS_FOLDER, img_name), img)


if __name__ == '__main__':
	imgs, img_names = load_imgs()
	imgs = resize(imgs)
	save_imgs(imgs, img_names)
