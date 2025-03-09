import glob
import json
import os
import random
from typing import List
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self,
                root_dir,
                is_train:bool=True,
                use_flip:bool=False,
                use_rotation:bool=False,
                use_colorJitter:bool=False,
                im_size:list=(224,224),
                max_num_imgs:int=None
                ):
        super(ImageDataset, self).__init__()
        assert isinstance(root_dir, str)
        assert isinstance(use_flip, bool)
        assert isinstance(use_rotation, bool)
        # assert isinstance(is_augment, bool)
        self.is_train = is_train
        self.root_dir = root_dir
        self.use_flip = use_flip
        self.use_rotation = use_rotation
        self.use_colorJitter = use_colorJitter
        self.is_augment = True if use_flip or use_rotation else False
        self.im_size = im_size

        # self.img_paths = self.find_imgs(self.root_dir)
        file_name = "train.txt" if is_train else "val.txt"
        with open(os.path.join(self.root_dir, file_name),'r',encoding="utf-8") as f:
            self.img_paths = [line.strip() for line in f.readlines()]

        if max_num_imgs is not None:
            self.img_paths = random.sample(self.img_paths, max_num_imgs)
        
            
        self.colorJitter = transforms.ColorJitter(0.5,0.5,0.5,0.5)

    def find_imgs(self, root_dir:str, ext:List[str]=('*.png', '*.jpg', '*.jpeg')):
        imgs = []
        for ext in ext:
            imgs.extend(glob.glob(os.path.join(root_dir, ext)))
        return imgs

    
    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = self.readImage(img_path)
        # if img is None:
        #     print("img error: {}".format(img_name))
        # print("img name: {}".format(img_name))
        if self.is_augment:
            img = augment(img, hflip=self.use_flip, rotation=self.use_rotation)


        # BGR to RGB
        # img = img[:, :, [2, 1, 0]]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.im_size is not None:
            img = cv2.resize(img, self.im_size, interpolation = cv2.INTER_AREA)
        # img_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2,1,0)))).float()
        # HWC to CHW
        img_np=np.transpose(img, (2,0,1))
        # numpy to tensor
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_np)).float().div(255)
        
        if self.use_colorJitter:
            img_tensor=self.colorJitter(img_tensor)
        


        return {'img': img_tensor, 'name': img_path}
    

    def readImage(self, filename):
        img=cv2.imdecode(np.fromfile(filename,dtype=np.uint8),-1)
        # img = cv2.imread(filename=filename)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img



    def __len__(self):
        return len(self.img_paths)



def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): (h, w, c)Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs
        
        



class FeatureDataset(Dataset):
    def __init__(self,
                root_dir,
                is_train:bool=True,
                aug_prob:float=0.0,
                seed:int=1234,
                ):
        super(FeatureDataset, self).__init__()
        assert isinstance(root_dir, str)

        self.rng = np.random.default_rng(seed)

        self.aug_prob = aug_prob
        self.is_train = is_train
        self.root_dir = root_dir

        labels_path = os.path.join(self.root_dir,"labels.json")
        self.labels:dict = None
        if not os.path.exists(labels_path):
            raise FileNotFoundError("labels.json not found in the root directory")
        
        with open(labels_path,'r',encoding="utf-8") as f:
            self.labels:dict = json.load(f)

        # self.feat_names = [os.path.basename(img_path)[:-4] for img_path in glob.glob(os.path.join(self.root_dir, "features","*.npy"))]
        with open(os.path.join(self.root_dir,"train_features.txt" if is_train else "val_features.txt"),'r',encoding="utf-8") as f:
            self.feat_names = [line.strip() for line in f.readlines()]

    
    def __getitem__(self, index):
        is_aug = True if self.rng.uniform() < self.aug_prob else False
        feat_name = self.feat_names[index]
        feat_path = os.path.join(self.root_dir,"features" if not is_aug else "aug_features",f"{feat_name}.npy")
        feat = torch.from_numpy(np.load(feat_path)).float()

        label = self.labels[feat_name]
        label=torch.tensor(label,dtype=torch.float32)



        return {'feat': feat, 'label': label, 'name': feat_name}
    

    def __len__(self):
        return len(self.feat_names)
    
