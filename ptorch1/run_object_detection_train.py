from pcore.deep_learn.classes_dict import ClassesDict
from pcore.ptorch.faster_rcnn import FasterRcnnNet
from pcore.ptorch.mini_trainer import MiniTrainer
from pcore.ptorch.object_detection_dataset import ObjectDetectionDataset

if __name__ == '__main__':
    classes_dict = ClassesDict()
    classes_dict.add_classes_names(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    images_dir = f"d:/demo/training/captchaImages"
    dataset = ObjectDetectionDataset(images_dir, classes_dict)

    model = FasterRcnnNet(classes_dict)
    trainer = MiniTrainer(model)
    trainer.batch_size = 2
    trainer.log_interval = 2
    trainer.fit_model(dataset)
