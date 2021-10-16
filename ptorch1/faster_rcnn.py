import torch
import torchvision
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

from pcore.deep_learn.classes_dict import ClassesDict
from pcore.ptorch.object_detection_dataset import ObjectDetectionDataset


def _is_iterable(variable):
    return isinstance(variable, list) or isinstance(variable, tuple)


def normalize_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def default_transforms():
    return transforms.Compose([transforms.ToTensor(), normalize_transform()])


class FasterRcnnNet(nn.Module):
    def __init__(self, classes_dict: ClassesDict):
        super().__init__()
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self._replace_pretrained_head(classes_dict)
        self._model.to(self._device)

    def start_train(self, checkpoints_dir):
        pass

    def train(self):
        self._model.train()

    def eval(self):
        pass

    def forward(self, x):
        return self._model(x)

    def train_step(self, images, targets):
        loss_dict = self._model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        return total_loss

    def validation_step(self, images, targets):
        return self.train_step(images, targets)

    def predict_step(self, images):
        self._model.eval()
        with torch.no_grad():
            if not _is_iterable(images):
                images = [images]
            perform_transform = default_transforms()
            images = [perform_transform(img) for img in images]

            images = [img.to(self._device) for img in images]
            predictions = self._model(images)
            predictions = [{k: v.to(torch.device('cpu')) for k, v in p.items()} for p in predictions]
            return predictions

    def _replace_pretrained_head(self, classes_dict):
        user_classes_len = classes_dict.get_size()
        in_features = self._model.roi_heads.box_predictor.cls_score.in_features
        # Replace the pre-trained head with a new one (note: +1 because of the __background__ class)
        self._model.roi_heads.box_predictor = FastRCNNPredictor(in_features, user_classes_len + 1)
        self._disable_normalize = False
        self._classes = ['__background__'] + classes_dict.names
        self._int_mapping = {label: index for index, label in enumerate(classes_dict.names)}

