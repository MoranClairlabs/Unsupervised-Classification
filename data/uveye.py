"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import pickle
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.mypath import MyPath
from torchvision.datasets.utils import check_integrity
import glob


class UVEYE(Dataset):
    def __init__(self, root=MyPath.db_root_dir('uveye'), train=True, transform=None, test=False,
                 download=False):

        super(UVEYE, self).__init__()
        self.root = '/content/drive/MyDrive/Unsupervised-Classification/data/uveye'
        self.transform = transform
        self.train = train  # training set or test set
        self.classes = [0, 1, 2, 3]#['screw/capsule', 'pill/meta_nut', 'meta_nut/pill', 'capsule/screw']
        self.class_dic = {'screw': 0, 'pill':1, 'metal_nut': 2, 'capsule': 3}
        self.black_white_dic = {'white': 0, 'black': 1}
        self.train_list = ['white', 'black']
        self.val_list = ['white', 'black']
        self.test_list = ['screw', 'pill', 'metal_nut', 'capsule']
        self.test = test
        self.targets = []
        self.black_or_white = []
        self.img_size = 128
        if self.train:
            self.base_folder = 'black_white_dataset_small_size/train'
            downloaded_list = self.train_list
        elif not(self.test):
            self.base_folder = 'black_white_dataset_small_size/test'
            downloaded_list = self.val_list
        else:
            self.base_folder = 'categories_dataset_small/test'
            downloaded_list = self.test_list

        self.data = []

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            images_paths = glob.glob(os.path.join(file_path, "*.png"))
            for img_path in images_paths:
                img = Image.open(img_path, 'r')
                img = img.convert('RGB')
                # img = img.resize((self.img_size,self.img_size))
                self.data.append(np.array(img.getdata()).reshape(img.size[0], img.size[1], 3))
                if self.test:
                    self.targets.append(self.class_dic[file_name])
                    self.black_or_white.append(file_name)
                else:
                    self.targets.append(self.black_white_dic[file_name])
                    self.black_or_white.append(file_name)

        self.data = np.vstack(self.data).reshape(-1, self.img_size, self.img_size, 3) # convert to HWC

        # self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        if self.targets is not None:
            img, label = self.data[index], self.targets[index]
            target = self.classes[label]
            class_name = self.classes[label]
        else:
            img, target = self.data[index], 255 # 255 is an ignore index
            class_name = 'unlabeled'
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img.astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'class_name': class_name}}

        return out

    def get_image(self, index):
        img = self.data[index]
        return img

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")