from math import degrees
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets
import logging
from torch.utils.data.sampler import Sampler

from torchvision.transforms.transforms import RandomChoice, RandomRotation
from utils.custom_data import CarsDataset, Cub2011

import cv2
np.random.seed(0)


class MPerClassSampler(Sampler):
    def __init__(self, indexes, labels, batch_size, m=4):
        self.labels = np.array(labels)
        self.labels_unique = np.unique(labels)
        self.batch_size = batch_size
        self.m = m
        self.indexes = np.array(indexes)
        assert batch_size % m == 0, 'batch size must be divided by m'

    def __len__(self):
        return len(self.labels) // self.batch_size

    def __iter__(self):
        for _ in range(self.__len__()):
            labels_in_batch = set()
            inds = np.array([], dtype=np.int)

            while inds.shape[0] < self.batch_size:
                sample_label = np.random.choice(self.labels_unique)
                if sample_label in labels_in_batch:
                    continue

                labels_in_batch.add(sample_label)
                sample_label_ids = np.argwhere(np.in1d(self.labels, sample_label)).reshape(-1)
               
                subsample = np.random.permutation(sample_label_ids)[:self.m]
                # print(self.labels[subsample])
                inds = np.append(inds, subsample)
               

            inds = inds[:self.batch_size]
            inds = np.random.permutation(inds)
            yield list(self.indexes[inds])


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=1, max=5):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 100% chance
        prob = np.random.random_sample()

        if prob < 1:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

np.random.seed(0)
logger = logging.getLogger(__name__)

class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return [xi, xj]

class OnlyOneDataTransform(object):
    def __init__(self, transform):
        self.transform = transform
        self.no_transform =  transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.no_transform(sample)
        return [xi, xj]

class DataTransform(object):
    def __init__(self, transform):
        self.transform = transform
       
    def __call__(self, sample):
        xi = self.transform(sample)
        return xi

class DataSetWrapper(object):

    def __init__(self, config):
        data_config = config['dataset']
        self.batch_size = config['batch_size']
        self.num_workers = data_config['num_workers']
        self.valid_size = data_config['valid_size'] 
        self.s = data_config['s']
        self.input_shape = eval(data_config['input_shape'])
        self.name = data_config['name']
        self.aug = data_config['augment']
        self.data_folder = data_config['data_folder']
        self.mperclass = data_config['mperclass']
        self.type = data_config['type']

        if(self.type == "2"):
            self.transform_type = OnlyOneDataTransform
        elif(self.type == "1"):
            self.transform_type = DataTransform
       

    def _get_simclr_pipeline_transform(self):
        
        if(self.aug == "Random"):
            color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
            data_transforms = transforms.Compose([
                                            transforms.RandomResizedCrop(size=(self.input_shape[0], self.input_shape[1]), scale=(0.5, 1.0)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=int(0.1 * self.input_shape[0]) + 1),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        elif(self.aug == "Gauss"):
            data_transforms = transforms.Compose([
                                            transforms.RandomResizedCrop(size=(self.input_shape[0], self.input_shape[1]), scale=(0.8, 1.0)),
                                            GaussianBlur(kernel_size=int(0.1 * self.input_shape[0]) + 1),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        else:
            data_transforms = transforms.Compose([
                transforms.Resize(size=(self.input_shape[0], self.input_shape[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        return data_transforms


    def get_train_validation_data_loaders(self):
       
        
        data_augment = self._get_simclr_pipeline_transform()
        
        if(self.name == "car"):
            self.train_dataset = CarsDataset(root= self.data_folder, 
                                    transform=self.transform_type(data_augment), train=True)
        elif(self.name == "cub"):
            self.train_dataset = Cub2011(root= self.data_folder, 
                                    transform=self.transform_type(data_augment), train=True)

        # obtain training indices that will be used for validation

        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        if(self.mperclass == 0) :
            # define samplers for obtaining training and validation batches
            train_sampler = SubsetRandomSampler(train_idx)
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        else:
            labels = [self.train_dataset.data.iloc[idx].target for idx in train_idx]
            
            train_sampler = MPerClassSampler(train_idx, labels, batch_size=self.batch_size, m=self.mperclass)
            train_loader = DataLoader(self.train_dataset, batch_sampler=train_sampler,
                                  num_workers=self.num_workers)

        valid_sampler = SubsetRandomSampler(valid_idx)

        

        valid_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)
        return train_loader, valid_loader

    def get_test_data_loaders(self):

        data_augment = transforms.Compose([
            transforms.Resize(size=(self.input_shape[0], self.input_shape[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

      
        if(self.name == "car"):
            self.test_dataset = CarsDataset(root= self.data_folder, train=False,
                                    transform=self.transform_type(data_augment))
        elif(self.name == "cub"):
            self.test_dataset = Cub2011(root= self.data_folder, train=False,
                                    transform=self.transform_type(data_augment))

        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=True)
 
class Multiresolution_DataSetWrapper(object):

    def __init__(self, config):
        data_config = config['dataset']
        self.batch_size = config['SimCLR']['batch_size']
        self.num_workers = data_config['num_workers']
        self.input_shape = eval(data_config['input_shape'])
        self.data_folder = data_config['data_folder']
       

    def _get_nmulti_resolution_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        data_transforms = transforms.Compose([
                                            transforms.Resize(size=(self.input_shape[0], self.input_shape[1])),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        return data_transforms


    def get_test_data_loaders(self):

        data_augment = self._get_nmulti_resolution_transform()

        self.test_dataset = Cub2011(root= self.data_folder, train=False,
                                    transform=MultiScaleTransform(data_augment))    

        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=True)


class MultiScaleTransform(object):
    def __init__(self, transform):
        self.transform1 = transforms.Resize(size=(32, 32))
        self.transform2 = transforms.Resize(size=(64, 64))
        self.transform3 = transforms.Resize(size=(128, 128))
        self.transform4 = transforms.Resize(size=(256, 256))
        self.transform = transform

    def __call__(self, sample):
        x1 = self.transform(self.transform1(sample))
        x2 = self.transform(self.transform2(sample))
        x3 = self.transform(self.transform3(sample))
        x4 = self.transform(self.transform4(sample))
        x5 = self.transform(sample)
        return [x1, x2, x3, x4, x5]






