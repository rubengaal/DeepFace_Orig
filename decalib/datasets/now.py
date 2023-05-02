
import os, sys

import face_alignment
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from PIL import Image
from matplotlib import pyplot as plt
from skimage import transform, io
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import utils


class NoWDataset(Dataset):
    def __init__(self, ring_elements=6, crop_size=224, scale=1.6): # scale 1.6
        folder = 'D:/OE/szakdolgozat/NoW'
        self.data_path = os.path.join(folder, 'imagepathsvalidation.txt')
        with open(self.data_path) as f:
            self.data_lines = f.readlines()

        self.imagefolder = os.path.join(folder, 'NoWDataset/NoW_Dataset/final_release_version', 'iphone_pictures')
        self.bbxfolder = os.path.join(folder, 'NoWDataset/NoW_Dataset/final_release_version', 'detected_face')

        # self.data_path = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/test_image_paths_ring_6_elements.npy'
        # self.imagepath = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/iphone_pictures/'
        # self.bbxpath = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/detected_face/'
        self.crop_size = crop_size
        self.scale = scale

        self.facedet = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')
        self.transform = transforms.Compose([Rescale(224), RandomCrop(224), ToTensor()])
    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, index):
        imagepath = os.path.join(self.imagefolder, self.data_lines[index].strip()) #+ '.jpg'
        bbx_path = os.path.join(self.bbxfolder, self.data_lines[index].strip().replace('.jpg', '.npy'))
        bbx_data = np.load(bbx_path, allow_pickle=True, encoding='latin1').item()
        # box = np.array([[bbx_data['left'], bbx_data['top']], [bbx_data['right'], bbx_data['bottom']]]).astype('float32')
        left = bbx_data['left']; right = bbx_data['right']
        top = bbx_data['top']; bottom = bbx_data['bottom']

        imagename = imagepath.split('/')[-1].split('.')[0]
        image = Image.open(imagepath)

        #h, w, _ = image.shape
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        size = int(old_size*self.scale)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.crop_size - 1], [self.crop_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        #image = image/255.
        #dst_image = warp(image, tform.inverse, output_shape=(self.crop_size, self.crop_size))
        #dst_image = dst_image.transpose(2, 0, 1)


        image = image.crop((left - 200, top - 200, right + 200, bottom + 200))
        image = image.resize((224, 224))
        image = np.array(image)
        image = image/255
        image = torch.from_numpy(image).type(dtype = torch.float32)
        image = image.permute(2,0,1)
        return {'image': image,
                'imagename': self.data_lines[index].strip().replace('.jpg', ''),
                # 'tform': tform,
                # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }

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

        img = transform.resize(image, (new_h, new_w))
        #msk = transform.resize(mask, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        sample = {'image': img, 'landmark': landmarks}

        return sample


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
        #cropped_mask = warp(mask, tform.inverse, output_shape=(224,224))
        # # change kpt accordingly
        cropped_kpt = np.dot(tform.params, np.hstack([landmarks, np.ones([landmarks.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        cropped_kpt[:, :2] = cropped_kpt[:, :2] / 224 * 2 - 1  # image_size

        return {'image': cropped_image, 'landmark': cropped_kpt}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmark']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image).type(dtype = torch.float32),
                'landmark': torch.from_numpy(landmarks).type(dtype = torch.float32)}


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch  = \
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
    print("Hello World")

    testdata = NoWDataset(scale=2.5) #1.6

    plt.figure()

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')

    images = testdata[0]['image'].cpu()

    #plt.imshow(images.cpu().permute(1,2,0))
    landmarks = fa.get_landmarks(images)
    landmarks = np.asarray(landmarks)
    landmarks = landmarks.astype('float').reshape(-1, 2)

    show_landmarks(images,landmarks)

    plt.show()