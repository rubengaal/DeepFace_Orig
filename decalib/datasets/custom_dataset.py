from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import estimate_transform, warp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from decalib.utils.config import cfg
from decalib.datasets import build_datasets
from decalib.datasets.CelebA import CelebDataset


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self,train,transform=None,test_mode=False,landmark_file=False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train = train
        self.test_mode = test_mode

        if self.train:
            csv_file = 'D:/OE/szakdolgozat/celeba/celeba/train_landmarks.csv'  # self.root + '../train_landmarks.csv'
        else:
            csv_file = 'D:/OE/szakdolgozat/celeba/celeba/val_landmarks.csv'  # self.root + '../val_landmarks.csv'
        if self.test_mode:
            self.train = False
            if landmark_file:
                csv_file = landmark_file
            else:
                csv_file = 'D:/OE/szakdolgozat/celeba/celeba/test_landmarks.csv'# self.root + '../test_landmarks.csv'
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = 'D:/OE/szakdolgozat/celeba/celeba/img_align_celeba/img_align_celeba/'
        self.transform = transform
        self.test_mode = test_mode
        self.train = train




    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        try:
            mask_name = os.path.join("D:/OE/szakdolgozat/celeba/celeba/visible_skin_masks",self.landmarks_frame.iloc[idx, 0]).replace('.jpg','_visible_skin_mask.png')
            mask = io.imread(mask_name)
        except:
            mask = image
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmark': landmarks, 'mask': mask}

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
        image, landmarks, mask = sample['image'], sample['landmark'], sample['mask']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        msk = transform.resize(mask, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmark': landmarks, 'mask': msk}


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
        image, landmarks, mask = sample['image'], sample['landmark'], sample['mask']

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
        cropped_mask = warp(mask, tform.inverse, output_shape=(224,224))
        # # change kpt accordingly
        cropped_kpt = np.dot(tform.params, np.hstack([landmarks, np.ones([landmarks.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        cropped_kpt[:, :2] = cropped_kpt[:, :2] / 224 * 2 - 1  # image_size

        return {'image': cropped_image, 'landmark': cropped_kpt, 'mask': cropped_mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks, mask = sample['image'], sample['landmark'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image).type(dtype = torch.float32),
                'landmark': torch.from_numpy(landmarks).type(dtype = torch.float32),
                'mask': torch.from_numpy(mask).type(dtype = torch.float32)}


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch , masks_batch = \
            sample_batched['image'], sample_batched['landmark'], sample_batched['mask']
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
    transformed_dataset = FaceLandmarksDataset(train=True,transform=transforms.Compose([Rescale(224),RandomCrop(224),ToTensor()]))

    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=0)

    train_iter = iter(dataloader)

    for x in range(3):
        sample_batched = next(train_iter)
        print(x, sample_batched['image'].size(),
              sample_batched['landmark'].size(),
              sample_batched['mask'].size())

        # observe 4th batch and stop.

        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break