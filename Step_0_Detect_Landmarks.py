import os

import face_alignment
import csv
from skimage import io
import pandas as pd
from tqdm import tqdm
import torch
'''------------------------------------
  Prepare Landmarks for Images
------------------------------------'''

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')

def detect_landmarks_celeba():
    root = 'D:/OE/szakdolgozat/celeba/celeba'

    dataset_path = root + '/img_align_celeba/img_align_celeba'
    root_folder = os.listdir(dataset_path)

    partitions_file = open(root + '/list_eval_partition.csv')
    lines = partitions_file.readlines()

    partition_dict = {}

    for line in tqdm(lines):
        data = line.strip().split(',')
        partition_dict.update({data[0]: data[1]})

    # results = pd.read_csv(root + '/train_landmarks.csv')
    # for filename in root_folder:
    #    if (results[results.columns[0]] == filename).any():
    #        root_folder.remove(filename)

    for filename in tqdm(root_folder):
        input_img = io.imread(dataset_path + '/' + filename)
        det = fa.get_landmarks_from_image(input_img)

        partition = partition_dict[filename].strip()

        if os.stat(root + '/train_landmarks.csv').st_size == 0:
            train_lmk_count = 0
        else:
            train_lmk_count = len(pd.read_csv(root + '/train_landmarks.csv'))

        if os.stat(root + '/test_landmarks.csv').st_size == 0:
            test_lmk_count = 0
        else:
            test_lmk_count = len(pd.read_csv(root + '/test_landmarks.csv'))

        # if os.stat(root + '/val_landmarks.csv').st_size == 0:
        #   val_lmk_count = len(pd.read_csv(root + '/val_landmarks.csv'))

        # if partition == '0' and train_lmk_count>19000: continue
        # elif partition == '1' and test_lmk_count>7500: continue
        # elif partition == '1' and val_lmk_count>7500: break
        # elif partition == '0' and train_lmk_count < 19000:

        if partition == '0':

            with open(root + '/train_landmarks.csv', encoding='utf-8', mode='a+', newline='') as f:
                writer = csv.writer(f)

                if det is None:
                    with open(root + '/noFace.csv', encoding='utf-8', mode='a+', newline='') as m:
                        writerm = csv.writer(m)
                        filet = []
                        filet.append(filename)
                        writerm.writerow(filet)
                else:
                    landmark = det[0].reshape(136).tolist()
                    landmark.insert(0, filename)
                    writer.writerow(landmark)
                print("train_landmarks")

        elif partition == '1':

            with open(root + '/test_landmarks.csv', encoding='utf-8', mode='a+', newline='') as f:
                writer = csv.writer(f)
                if det is None:
                    with open(root + '/noFace.csv', encoding='utf-8', mode='a+', newline='') as m:
                        writerm = csv.writer(m)
                        filet = []
                        filet.append(filename)
                        writerm.writerow(filet)
                else:
                    landmark = det[0].reshape(136).tolist()
                    landmark.insert(0, filename)
                    writer.writerow(landmark)
                print("test_landmarks")

        elif partition == '2':

            with open(root + '/val_landmarks.csv', encoding='utf-8', mode='a+', newline='') as f:
                writer = csv.writer(f)
                if det is None:
                    with open(root + '/noFace.csv', encoding='utf-8', mode='a+', newline='') as m:
                        writerm = csv.writer(m)
                        filet = []
                        filet.append(filename)
                        writerm.writerow(filet)
                else:
                    landmark = det[0].reshape(136).tolist()
                    landmark.insert(0, filename)
                    writer.writerow(landmark)
            print("val_landmarks")
            val_lmk_count = 0

def detect_landmarks_NoW():
    root = 'D:/OE/szakdolgozat/celeba/celeba'

    dataset_path = root + '/img_align_celeba/img_align_celeba'
    root_folder = os.listdir(dataset_path)

    partitions_file = open(root + '/list_eval_partition.csv')
    lines = partitions_file.readlines()

    partition_dict = {}

    for line in tqdm(lines):
        data = line.strip().split(',')
        partition_dict.update({data[0]: data[1]})

    # results = pd.read_csv(root + '/train_landmarks.csv')
    # for filename in root_folder:
    #    if (results[results.columns[0]] == filename).any():
    #        root_folder.remove(filename)

    for filename in tqdm(root_folder):
        input_img = io.imread(dataset_path + '/' + filename)
        det = fa.get_landmarks_from_image(input_img)

        partition = partition_dict[filename].strip()

        if os.stat(root + '/train_landmarks.csv').st_size == 0:
            train_lmk_count = 0
        else:
            train_lmk_count = len(pd.read_csv(root + '/train_landmarks.csv'))

        if os.stat(root + '/test_landmarks.csv').st_size == 0:
            test_lmk_count = 0
        else:
            test_lmk_count = len(pd.read_csv(root + '/test_landmarks.csv'))

        # if os.stat(root + '/val_landmarks.csv').st_size == 0:
        #   val_lmk_count = len(pd.read_csv(root + '/val_landmarks.csv'))

        # if partition == '0' and train_lmk_count>19000: continue
        # elif partition == '1' and test_lmk_count>7500: continue
        # elif partition == '1' and val_lmk_count>7500: break
        # elif partition == '0' and train_lmk_count < 19000:

        if partition == '0':

            with open(root + '/train_landmarks.csv', encoding='utf-8', mode='a+', newline='') as f:
                writer = csv.writer(f)

                if det is None:
                    with open(root + '/noFace.csv', encoding='utf-8', mode='a+', newline='') as m:
                        writerm = csv.writer(m)
                        filet = []
                        filet.append(filename)
                        writerm.writerow(filet)
                else:
                    landmark = det[0].reshape(136).tolist()
                    landmark.insert(0, filename)
                    writer.writerow(landmark)
                print("train_landmarks")

        elif partition == '1':

            with open(root + '/test_landmarks.csv', encoding='utf-8', mode='a+', newline='') as f:
                writer = csv.writer(f)
                if det is None:
                    with open(root + '/noFace.csv', encoding='utf-8', mode='a+', newline='') as m:
                        writerm = csv.writer(m)
                        filet = []
                        filet.append(filename)
                        writerm.writerow(filet)
                else:
                    landmark = det[0].reshape(136).tolist()
                    landmark.insert(0, filename)
                    writer.writerow(landmark)
                print("test_landmarks")

        elif partition == '2':

            with open(root + '/val_landmarks.csv', encoding='utf-8', mode='a+', newline='') as f:
                writer = csv.writer(f)
                if det is None:
                    with open(root + '/noFace.csv', encoding='utf-8', mode='a+', newline='') as m:
                        writerm = csv.writer(m)
                        filet = []
                        filet.append(filename)
                        writerm.writerow(filet)
                else:
                    landmark = det[0].reshape(136).tolist()
                    landmark.insert(0, filename)
                    writer.writerow(landmark)
            print("val_landmarks")
            val_lmk_count = 0





