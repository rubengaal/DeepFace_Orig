# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys

import face_alignment
import pandas as pd
import torch
import torchvision.transforms.functional as fn
import torchvision.transforms
from matplotlib import pyplot as plt
from skimage import transform, io
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
import scipy.io
from torchvision import utils

from decalib.datasets import detectors


def video2sequence(video_path, sample_step=10):
    videofolder = os.path.splitext(video_path)[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        # if count%sample_step == 0:
        imagepath = os.path.join(videofolder, f'{video_name}_frame{count:04d}.jpg')
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list

class TestData(Dataset):

    def __init__(self, transform=None):

        self.root_dir = "D:/Dev/repos/deepcheap/DeepFace_Orig/TestSamples/examples"

        self.imagepath_list = glob(self.root_dir + '/*.jpg') + glob(self.root_dir + '/*.png') + glob(self.root_dir + '/*.bmp')
        self.imagepath_list = sorted(self.imagepath_list)
        self.transform = transform
        self.facedet = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')


    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imagepath = self.imagepath_list[idx]
        imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
        image = io.imread(imagepath)


        landmarks = self.facedet.get_landmarks_from_image(image)
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmark': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmark']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image,  (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmark': landmarks}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        self.scale = [1.4, 1.8]
        self.trans_scale = 0.  # 0.5?
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmark']

        # h, w = image.shape[:2]
        # new_h, new_w = self.output_size
        #
        # top = np.random.randint(0, h - new_h)
        # left = np.random.randint(0, w - new_w)
        #
        # image = image[top: top + new_h,
        #               left: left + new_w]
        #
        # landmarks = landmarks - [left, top]

        left = np.min(landmarks[:, 0]);
        right = np.max(landmarks[:, 0]);
        top = np.min(landmarks[:, 1]);
        bottom = np.max(landmarks[:, 1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
        trans_scale = (np.random.rand(2) * 2 - 1) * self.trans_scale
        center = center + trans_scale * old_size  # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]

        src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])
        DST_PTS = np.array([[0, 0], [0, 224 - 1], [224 - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        cropped_image = warp(image, tform.inverse, output_shape=(224, 224))
        # # change kpt accordingly
        cropped_kpt = np.dot(tform.params, np.hstack([landmarks, np.ones([landmarks.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        #cropped_kpt[:, :2] = cropped_kpt[:, :2] / 224 * 2 - 1  # image_size

        return {'image': cropped_image, 'landmark': cropped_kpt}




class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmark = sample['image'], sample['landmark']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image).type(dtype=torch.float32),
                'landmark': torch.from_numpy(landmark).type(dtype=torch.float32),
                }

class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmark = sample['image'], sample['landmark']

        mean = [0.5063, 0.4258, 0.3837]
        std = [0.2676, 0.2565, 0.2627]

        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std

        return {'image': image,
                'landmark': landmark,
                }


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
        sample_batched['image'], sample_batched['landmark']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')


if __name__ == '__main__':
    transformed_dataset = TestData( transform=transforms.Compose(
        [Normalize(), Rescale(224), RandomCrop(224), ToTensor()]))

    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=0, drop_last=True)

    train_iter = iter(dataloader)


    sample_batched = next(train_iter)
    print(sample_batched['image'].size(),
          sample_batched['landmark'].size())
    # observe 4th batch and stop.
    plt.figure()
    show_landmarks_batch(sample_batched)
    plt.axis('off')
    plt.ioff()
    plt.show()
