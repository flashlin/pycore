import glob
import os

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, SubsetRandomSampler
from torchvision import transforms

from deep_learn.classes_dict import ClassesDict
from common.io import info


def read_image_classification_data(image_path, image_channels=1):
    image_resize = (28, 28)
    means = [0.13499917, 0.13499917, 0.13499917]
    stdevs = [0.29748289, 0.29748289, 0.29748289]

    transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(means, stdevs),
        transforms.Resize(image_resize),
    ])

    if image_channels == 1:
        transform1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize([0.5], [0.5]),
            transforms.Resize(image_resize),
        ])

    img = Image.open(image_path).convert('RGB')
    img = transform1(img)
    img = img[None]
    return img


class ImageClassificationDataset(Dataset):
    batch_size = 33  # 32-512
    validation_split = 0.1
    shuffle_dataset = True
    random_seed = 42
    image_channels = 1
    image_resize = (28, 28)

    def __init__(self, images_dir, classes_dict: ClassesDict):
        super(ImageClassificationDataset, self).__init__()
        self.classes_dict = classes_dict
        self.images_dir = images_dir
        self.images = self.get_images_list()
        self.device = "gpu" if torch.cuda.is_available is True else "cpu"
        info(f"images count = {len(self.images)}")
        self.means = [0.13499917, 0.13499917, 0.13499917]
        self.stdevs = [0.29748289, 0.29748289, 0.29748289]
        if self.image_channels == 1:
            self.means = [0.13500238, 0.13500238, 0.13500238]
            self.stdevs = [0.29748997, 0.29748997, 0.29748997]

    def compute_mean_std(self):
        """
        計算數據集的均值和標準差
        : return :
        """
        info(f"compute mean std...")
        dataset = self.__get_all_images()
        num_imgs = len(self.images)
        for data in dataset:
            data = np.asarray(data) / 255.0
            mean = np.mean(data, axis=(0, 1))
            std = np.std(data, axis=(0, 1))
            self.means += mean
            self.stdevs += std
            # img = data[0]
            # for i in range(3):
            #     # 一個通道的均值和標準差
            #     self.means[i] += img[i, :, :].mean()
            #     self.stdevs[i] += img[i, :, :].std()
        self.means = np.asarray(self.means) / num_imgs
        self.stdevs = np.asarray(self.stdevs) / num_imgs
        return self.means, self.stdevs

    def get_train_validation_loaders(self):
        train_dataset = self
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.validation_split * dataset_size))
        if self.shuffle_dataset:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        info(f"train size={len(train_indices)} val size={len(val_indices)}")

        train_loader = self.__get_data_loader(train_indices)
        validation_loader = self.__get_data_loader(val_indices)
        return train_loader, validation_loader

    def __get_data_loader(self, indices):
        sampler = SubsetRandomSampler(indices)
        dataset = self
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.batch_size,
                                                  sampler=sampler)
        return data_loader

    # def get_means(self):
    #     train_data_loader, val_data_loader = self.get_train_validation_loader()
    #     return (train_data_loader.data / 255.0).mean(axis=(0, 1, 2))

    def get_images_list(self):
        all_image_files = []
        for classes_name in os.listdir(self.images_dir):
            image_files = glob.glob(f"{self.images_dir}/{classes_name}/*.jpg")
            self.classes_dict.add_classes_name(classes_name)
            for image_file in image_files:
                all_image_files.append(image_file)
        return all_image_files

    def save_classes_names(self, save_file_path):
        self.classes_dict.save(save_file_path)

    def __get_all_images(self):
        for img_path in self.images:
            # transform1 = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Grayscale(num_output_channels=1),
            #     transforms.Resize(self.image_resize),
            # ])
            img = Image.open(img_path).convert('RGB')
            if self.image_channels == 1:
                img = ImageOps.grayscale(img)
            # img = transform1(img)
            yield img

    def __getitem__(self, index):
        img_path = self.images[index]
        transform1 = transforms.Compose([
            # transforms.Grayscale(num_output_channels=1),
            # transforms.Normalize([0.5], [0.5]),
            transforms.Resize(self.image_resize),
            transforms.ToTensor(),
            transforms.Normalize(self.means, self.stdevs),
        ])
        if self.image_channels == 1:
            transform1 = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                # transforms.Normalize(self.means[0], self.stdevs[0]),
                transforms.Resize(self.image_resize),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        img = Image.open(img_path).convert('RGB')
        img = transform1(img)
        classes_name = os.path.dirname(img_path)
        classes_name = os.path.basename(classes_name)
        classes_id = self.classes_dict.add_classes_name(classes_name)
        # if self.transform is not None:
        #     img = self.transform(img)
        # print(f"id={classes_id} name={classes_name}")
        # label = [0.0 for i in range(10)]
        # label[classes_id - 1] = 1.0
        # img = img.to(self.device, torch.float)
        classes_id = classes_id - 1
        return img, classes_id

    def __len__(self):
        return len(self.images)
