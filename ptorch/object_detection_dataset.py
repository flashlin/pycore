import os.path
from glob import glob

import torch
from torch.utils.data import Dataset
import random

from common.io import get_file_name, read_image
from deep_learn.annotation_xml import AnnotationXml
from deep_learn.classes_dict import ClassesDict
import pandas as pd
from torchvision import transforms


def default_transforms():
    return transforms.Compose([transforms.ToTensor(), normalize_transform()])


def normalize_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class ObjectDetectionDataset(Dataset):
    def __init__(self, image_folder, classes_dict: ClassesDict):
        self._image_folder = image_folder
        self._classes_dict = classes_dict
        self.transform = default_transforms()
        self._csv = ObjectDetectionDataset.xml_to_csv(image_folder, classes_dict)

    @staticmethod
    def xml_to_csv(image_folder, classes_dict):
        csv_rows = []
        image_id = 0
        for xml_file in glob(f"{image_folder}/*.xml"):
            image_name = get_file_name(xml_file)
            if not os.path.isfile(f"{image_folder}/{image_name}.jpg"):
                continue
            for annotation in AnnotationXml(xml_file):
                filename, width, height, label_name, (x_min, y_min, x_max, y_max) = annotation
                class_id = classes_dict.add_classes_name(label_name)
                csv_rows.append((
                    filename,
                    width, height,
                    class_id,
                    x_min, y_min, x_max, y_max,
                    image_id
                ))
                image_id += 1
        column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'image_id']
        csv = pd.DataFrame(csv_rows, columns=column_names)
        return csv

    def __len__(self):
        return len(self._csv['image_id'].unique().tolist())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        object_entries = self._csv.loc[self._csv['image_id'] == idx]

        img_name = os.path.join(self._image_folder, object_entries.iloc[0, 0])
        image = read_image(img_name)

        boxes = []
        labels = []
        for object_idx, row in object_entries.iterrows():
            # Read in xmin, ymin, xmax, and ymax
            box = self._csv.iloc[object_idx, 4:8]
            boxes.append(box)
            label = self._csv.iloc[object_idx, 3]
            labels.append(label)

        boxes = torch.tensor(boxes).view(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)

        targets = {'boxes': boxes, 'labels': labels}

        image = self.perform_transforms(image, object_entries, targets)

        return image, targets

    def perform_transforms(self, image, object_entries, targets):
        if not self.transform:
            return image

        width = object_entries.iloc[0, 1]
        height = object_entries.iloc[0, 2]
        # Apply the transforms manually to be able to deal with
        # transforms like Resize or RandomHorizontalFlip
        updated_transforms = []
        scale_factor = 1.0
        random_flip = 0.0
        for t in self.transform.transforms:
            # Add each transformation to our list
            updated_transforms.append(t)

            # If a resize transformation exists, scale down the coordinates
            # of the box by the same amount as the resize
            if isinstance(t, transforms.Resize):
                original_size = min(height, width)
                scale_factor = original_size / t.size

            # If a horizontal flip transformation exists, get its probability
            # so we can apply it manually to both the image and the boxes.
            elif isinstance(t, transforms.RandomHorizontalFlip):
                random_flip = t.p
        # Apply each transformation manually
        for t in updated_transforms:
            # Handle the horizontal flip case, where we need to apply
            # the transformation to both the image and the box labels
            if isinstance(t, transforms.RandomHorizontalFlip):
                if random.random() < random_flip:
                    image = transforms.RandomHorizontalFlip(1)(image)
                    for idx, box in enumerate(targets['boxes']):
                        # Flip box's x-coordinates
                        box[0] = width - box[0]
                        box[2] = width - box[2]
                        box[[0, 2]] = box[[2, 0]]
                        targets['boxes'][idx] = box
            else:
                image = t(image)
        # Scale down box if necessary
        if scale_factor != 1.0:
            for idx, box in enumerate(targets['boxes']):
                box = (box / scale_factor).long()
                targets['boxes'][idx] = box
        return image


class BatchDataLoader(torch.utils.data.DataLoader):

    def __init__(self, dataset, **kwargs):
        """
            >>> dataset = Dataset('images/')
            >>> loader = DataLoader(dataset, batch_size=2, shuffle=True)
            >>> for images, targets in loader:
            >>>     print(images[0].shape)
            >>>     print(targets[0])
            torch.Size([3, 1080, 1720])
            {'boxes': tensor([[884, 387, 937, 784]]), 'labels': ['person']}
            torch.Size([3, 1080, 1720])
            {'boxes': tensor([[   1,  410, 1657, 1079]]), 'labels': ['car']}
            ...
        """
        super().__init__(dataset, collate_fn=BatchDataLoader.collate_data, **kwargs)

    @staticmethod
    def collate_data(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)
