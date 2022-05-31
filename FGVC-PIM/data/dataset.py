import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from timm.data.transforms import _pil_interp, RandomResizedCropAndInterpolation
# from timm.data import transforms
import torchvision 

from PIL import Image
import copy
import torch
import pandas as pd
from sklearn.utils import shuffle
from .randaug import RandAugment


def build_loader(args):
    train_set, train_loader = None, None
    if args.train_root is not None:
        train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
        data_train = pd.read_csv("/home/data1/lkd/CVPR_FUNGI/Fungi_code/EDA/all_data.csv",header=0)
        data_train = shuffle(data_train,random_state=0)
        # num_classes = 1572
        # data_train = pd.read_csv("train_sample50.csv",header=0)
        samples_train = [{'path':data_train.file_dir[index],'label': data_train.classid[index]} for index in range(len(data_train))]
        train_set.data_infos = samples_train
        train_loader = torch.utils.data.DataLoader(train_set, num_workers=args.num_workers, shuffle=True, batch_size=args.batch_size)

    val_set, val_loader = None, None
    if args.val_root is not None:
        val_set = ImageDataset(istrain=False, root=args.val_root, data_size=args.data_size, return_index=True)
        data_val = pd.read_csv("/home/data1/lkd/CVPR_FUNGI/Fungi_code/EDA/all_data.csv",nrows=10000,header=0)
        data_val = shuffle(data_val,random_state=0)
        # num_classes = 1572
        # data_train = pd.read_csv("train_sample50.csv",header=0)
        samples_val = [{'path':data_val.file_dir[index],'label': data_val.classid[index]} for index in range(len(data_val))]
        val_set.data_infos = samples_val
        val_loader = torch.utils.data.DataLoader(val_set, num_workers=1, shuffle=True, batch_size=args.batch_size)

    return train_loader, val_loader

def get_dataset(args):
    if args.train_root is not None:
        train_set = ImageDataset(istrain=True, root=args.train_root, data_size=args.data_size, return_index=True)
        return train_set
    return None


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 istrain: bool,
                 root: str,
                 data_size: int,
                 return_index: bool = False):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.root = root
        self.data_size = data_size
        self.return_index = return_index

        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

        # 448:600
        # 384:510
        # 768:
        if istrain:
            # transforms.RandomApply([RandAugment(n=2, m=3, img_size=data_size)], p=0.1)
            # RandAugment(n=2, m=3, img_size=sub_data_size)
            self.transforms = transforms.Compose([
                        # transforms.RandomResizedCropAndInterpolation(size=(data_size, data_size), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=bilinear bicubic),
                        # transforms.RandomCrop((data_size, data_size)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.ColorJitter(brightness=[0.6, 1.4], contrast=[0.6, 1.4], saturation=[0.6, 1.4], hue=0.5),
                        RandomResizedCropAndInterpolation(size=(data_size, data_size),scale=(0.08, 1.0),ratio=(0.75, 1.3333),interpolation='bilinear bicubic'),
                        # transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        # transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((data_size, data_size), Image.BILINEAR),
                        transforms.CenterCrop((data_size, data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)


    def getDataInfo(self, root):
        data_infos = []
        folders = os.listdir(root)
        folders.sort() # sort by alphabet
        print("[dataset] class number:", len(folders))
        for class_id, folder in enumerate(folders):
            files = os.listdir(root+folder)
            for file in files:
                data_path = root+folder+"/"+file
                data_infos.append({"path":data_path, "label":class_id})
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1] # BGR to RGB.
        
        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)
        
        if self.return_index:
            # return index, img, sub_imgs, label, sub_boundarys
            return index, img, label
        
        # return img, sub_imgs, label, sub_boundarys
        return img, label
