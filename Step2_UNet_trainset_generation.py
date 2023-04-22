import numpy
import os
from tqdm import tqdm
import cv2
from face_toolbox_keras.models.parser import face_parser
import tensorflow as tf


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
def moreThan100k(img_name):
    number = int(img_name.lstrip('0').replace('.jpg',''))
    return number > 100000

def generateMasks():

    root = 'D:/OE/szakdolgozat/celeba/celeba/img_align_celeba/img_align_celeba/'
    dir_list = os.listdir(root)
    dir_list = [img_name for img_name in dir_list if moreThan100k(img_name)]

    for img in tqdm(dir_list):
        im = cv2.imread(root + img)[..., ::-1]
        im = cv2.resize(im, (224, 224))
        fp = face_parser.FaceParser()
        parsing_map = fp.parse_face(im, bounding_box=None, with_detection=False)[0]
        img_mask = numpy.array(parsing_map)
        img_mask[img_mask == 1] = 1
        img_mask[img_mask == 2] = 1
        img_mask[img_mask == 3] = 1
        img_mask[img_mask == 4] = 1
        img_mask[img_mask == 5] = 1
        img_mask[img_mask == 10] = 1
        img_mask[img_mask == 11] = 1
        img_mask[img_mask == 12] = 1
        img_mask[img_mask == 13] = 1
        img_mask[img_mask == 6] = 0
        img_mask[img_mask == 7] = 0
        img_mask[img_mask == 8] = 0
        img_mask[img_mask == 9] = 0
        img_mask[img_mask == 14] = 0
        img_mask[img_mask == 15] = 0
        img_mask[img_mask == 16] = 0
        img_mask[img_mask == 17] = 0
        img_mask[img_mask == 18] = 0

        cv2.imwrite('D:/OE/szakdolgozat/celeba/celeba/visible_skin_masks/' + img.replace('.jpg', '_visible_skin_mask.png'),
                    img_mask * 255)
    print(f'-- please check the results in D:/OE/szakdolgozat/celeba/celeba/')



