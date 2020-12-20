import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import os
import shutil
import csv
import json
import PIL.Image


class boundingBox:
    "This is a bounding box class"

    def __init__(self, edges):
        self.xmin = edges[0]
        self.ymin = edges[1]
        self.width = edges[2]
        self.height = edges[3]
        self.color = edges[4]
        self.edges = edges  # return array of all data


class imgBoxes:
    "This is img with its bounding boxes class"

    def __init__(self, imgname, boxes):
        self.imgname = imgname
        self.boxes = boxes  # this can be an array of boundingBox objects

    # print values of boxes for this img
    def print(self):
        print(self.imgname + ':')
        for box in self.boxes:
            print(box.edges, end=',')
        print('')


def read_imgs_annotations(filename):
    """"reads the annotation file into a list of object of type imgBoxes. """
    file = open(filename, 'r')
    lines = file.readlines()
    imgboxes = []  # this will be a list containing imgBoxes objects for all imgs
    for line in lines:
        imgname, boxesdata = line.split(':')
        boundingboxes = []
        boxes = boxesdata[1:-2].split('],[')  # list of strings containing box data
        for box in boxes:
            box_numeric = np.fromstring(box, sep=',', dtype=int)
            boundingbox = boundingBox(box_numeric)  # bounding box object from string
            boundingboxes.append(boundingbox)  # for each img we create list of boxes
        imgboxes.append(imgBoxes(imgname, boundingboxes))
    return imgboxes


def split_to_train_test(imgboxes):
    """"split images to train ant test set,and move them to respective folders"""
    imgs_train, imgs_test = train_test_split(imgboxes, test_size=0.33)
    path = os.getcwd()
    # create test and train imgs folder
    try:
        os.mkdir(path + os.sep+'test_images')
        os.mkdir(path + os.sep+ 'train_images')
    except OSError:
        pass

        for img in imgs_train:
            try:
                os.rename(path + os.sep+ 'busesTrain' +os.sep + img.imgname, path + os.sep+ 'train_images' +os.sep+ img.imgname)
            except FileNotFoundError:
                pass
        for img in imgs_test:
            try:
                os.rename(path + os.sep+ 'busesTrain' +os.sep + img.imgname, path + os.sep+ 'test_images' +os.sep+ img.imgname)
            except FileNotFoundError:
                pass
    return imgs_train, imgs_test


def move_files_to_main_dir(src_dir, target_dir):
    file_names = os.listdir(src_dir)
    for file_name in file_names:
        shutil.move(os.path.join(src_dir, file_name), target_dir)


def combine_train_test():
    """"move images from train and test folders back to main folder"""
    source_dir1 = os.getcwd() +os.sep+ 'train_images'+os.sep
    source_dir2 = os.getcwd() +os.sep+ 'test_images' +os.sep
    target_dir = os.getcwd() +os.sep+ 'busesTrain' +os.sep
    move_files_to_main_dir(source_dir1, target_dir)
    move_files_to_main_dir(source_dir2, target_dir)


def generate_train_csv(imgs_train):
    """"generate txt file of the format filepath,x1,y1,x2,y2,class_name"""
    dir_train = os.getcwd() + os.sep + 'train_images'
    f = open("annotationsTrainNew.txt", "w")
    f = open("annotationsTrainNew.txt", "a")
    for img in imgs_train:
        filename = dir_train + img.imgname
        for bus in img.boxes:
            f.write("%s %d,%d,%d,%d,%d\n" % (
            filename, bus.xmin, bus.ymin, bus.xmin + bus.width, bus.ymin + bus.height, bus.color))
    return os.getcwd() +os.sep+ 'annotationsTrainNew.txt'


#
# def my_dataset_function(filename,train_dir):
#     """"reads the annotation file into a list of object of type imgBoxes. """
#     file = open(filename, 'r')
#     lines = file.readlines()
#     imgdict={
#         "file_name":"",
#         "height": "",
#         "width": "",
#         "image_id": "",
#         "annotations": "",
#     }
#
#     annotations_dict={
#         "bbox": "",
#         "bbox_mode": 1,
#         "category_id": "",
#     }
#
#     imgboxes = []  # this will be a list containing imgBoxes objects for all imgs
#     for line in lines:
#         imgdict = {
#             "file_name": "",
#             "height": "",
#             "width": "",
#             "image_id": "",
#             "annotations": "",
#         }
#         imgname, boxesdata = line.split(':')
#         img = PIL.Image.open(train_dir+os.sep+imgname)
#         exif_data = img._getexif()
#         print(exif_data)
#         imgdict["width"]=exif_data[0xa002]
#         imgdict["height"]=exif_data[0xa003]
#         imgdict["file_name"]=train_dir+os.sep+imgname
#         imgdict["image_id"]=imgname
#
#         boundingboxes = []
#         boxes = boxesdata[1:-2].split('],[')  # list of strings containing box data
#         for box in boxes:
#             box_numeric = np.fromstring(box, sep=',', dtype=int)
#             boundingbox = boundingBox(box_numeric)  # bounding box object from string
#             boundingboxes.append(boundingbox)  # for each img we create list of boxes
#         imgboxes.append(imgBoxes(imgname, boundingboxes))
#     return list[dict]
#

# from detectron2.data import DatasetCatalog
# DatasetCatalog.register("my_dataset", my_dataset_function)
# later, to access the data:
# data: List[Dict] = DatasetCatalog.get("my_dataset")


imgboxes = read_imgs_annotations('annotationsTrain.txt')
imgs_train, imgs_test = split_to_train_test(imgboxes)
#combine_train_test()
csvFilePath = generate_train_csv(imgs_train)

# my_dataset_function('annotationsTrain.txt',os.getcwd()+os.sep+'busesTrain')
