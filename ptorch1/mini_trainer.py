import os

import numpy as np
from tqdm import tqdm
import torch
import re

from ptorch.dataset_utils import get_train_validation_loaders, get_train_validation_datasets
from common.io import info, get_folder_list, confirm_dirs
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from ptorch.object_detection_dataset import BatchDataLoader


class MiniTrainer:
    checkpoints_dir = "d:/demo/training/checkpoints"
    batch_size = 32
    max_epochs = 5
    log_interval = 10

    def __init__(self, nn_model):
        super().__init__()
        self.cuda_available = torch.cuda.is_available
        self.device = "gpu" if self.cuda_available is True else "cpu"
        info(f"device={self.device}")
        self.nn_model = nn_model
        self.model_name = type(nn_model).__name__

    def save_model_state(self, pt_path):
        confirm_dirs(os.path.dirname(pt_path))
        torch.save(self.nn_model.state_dict(), pt_path)

    def load_model_state(self, pt_path):
        self.nn_model.load_state_dict(torch.load(pt_path))

    def fit_model(self, dataset):
        self.save_pretrain_model_data(f"{self.checkpoints_dir}")
        self.load_pretrained_model()
        self.train_start(dataset)

    def save_pretrain_model_data(self, save_dir):
        if not hasattr(self.nn_model, 'start_train'):
            return
        self.nn_model.start_train(f"{save_dir}")

    def load_pretrained_model(self):
        pretrained_filename = self.get_last_checkpoints_filename()
        if pretrained_filename is not None and os.path.isfile(pretrained_filename):
            info(f"Found pretrained model at {pretrained_filename}, loading...")
            self.load_model_state(pretrained_filename)

    def get_last_checkpoints_filename(self):
        version_dirs = self.get_checkpoints_version_dirs()
        if len(version_dirs) == 0:
            return None
        for version_dir in version_dirs:
            pt_file_path = f"{version_dir}/last.pt"
            if os.path.isfile(pt_file_path):
                return pt_file_path
        return None

    def get_checkpoints_version_dirs(self):
        version_dirs = sorted(get_folder_list(f"{self.checkpoints_dir}/{self.model_name}"), reverse=True)
        return version_dirs

    def get_last_checkpoints_dir(self):
        base_version_dir = f"{self.checkpoints_dir}/{self.model_name}/version_"
        version_dirs = self.get_checkpoints_version_dirs()
        if len(version_dirs) == 0:
            return f"{base_version_dir}_0", base_version_dir, 0
        last_version_dir_name = os.path.basename(version_dirs[0])
        regex = re.compile(r'version_(\d+)')
        match = regex.search(last_version_dir_name)
        if match is None:
            return f"{base_version_dir}_0", base_version_dir, 0
        last_num = int(match.group(1))
        return f"{self.checkpoints_dir}/{self.model_name}/version_{last_num}", base_version_dir, last_num

    def save_training_model_state(self, epoch, loss):
        last_version_dir, base_version_dir, last_version_num = self.get_last_checkpoints_dir()
        pt_path = self.get_version_pt_path(last_version_num, epoch, loss)
        # info(f"save {pt_path}")
        self.save_model_state(pt_path)

    def save_last_model_state(self):
        last_version_dir, base_version_dir, last_version_num = self.get_last_checkpoints_dir()
        pt_path = f"{last_version_dir}/last.pt"
        self.save_model_state(pt_path)

    def get_version_pt_path(self, version_num, epoch, loss):
        version_dir = f"{self.checkpoints_dir}/{self.model_name}/version_{version_num}"
        pt_path = f"{version_dir}/epoch{epoch}-{loss}.pt"
        return pt_path

    def train_start(self, dataset):
        self.create_start_training_checkpoints_dir()
        self.min_valid_loss = np.inf

        # train_data_loader, val_data_loader = get_train_validation_loaders(dataset, batch_size=self.batch_size)

        dataset_train, dataset_validation = get_train_validation_datasets(dataset, batch_size=self.batch_size)
        train_data_loader = BatchDataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        val_data_loader = BatchDataLoader(dataset_validation, batch_size=self.batch_size, shuffle=True)
        # loader = DataLoader(dataset, batch_size=2, shuffle=True)

        last_version_dir, base_version_dir, last_version_num = self.get_last_checkpoints_dir()

        # model = self.nn_model
        # info(model)

        info("train start...")
        summary_writer = SummaryWriter(log_dir=f"{last_version_dir}/summary")
        for epoch in range(self.max_epochs):
            training_writer = SummaryWriter(log_dir=f"{last_version_dir}/epoch-{epoch}")
            self.train_step(train_data_loader, epoch, training_writer, summary_writer)
            self.val_step(val_data_loader, epoch, training_writer, summary_writer)
            training_writer.close()
        summary_writer.close()

    def create_start_training_checkpoints_dir(self):
        last_version_dir, base_version_dir, last_version_num = self.get_last_checkpoints_dir()
        confirm_dirs(f"{base_version_dir}{last_version_num + 1}")

    def configure_optimizer(self):
        model = self.nn_model
        if hasattr(model, 'optimizer'):
            return model.optimizer
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=1e-4, last_epoch=-1)
        return optimizer

    def train_step(self, train_loader, epoch, training_writer, summary_writer):
        nn_model = self.nn_model
        optimizer = self.configure_optimizer()
        running_loss = 0.0
        train_step_count = 0
        nn_model.train()
        for batch_idx, sample in enumerate(train_loader):
            # print(f"sample = {sample}")
            # inputs, labels = sample['data'], sample['target']
            inputs, labels = sample

            # info(f"labels = {labels}")
            # inputs, labels = self._to_device(inputs, labels)
            # for data, target in tqdm(train_loader_iter):
            optimizer.zero_grad()

            loss = nn_model.train_step(inputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss
            training_writer.add_scalar("batch/train loss", loss, batch_idx)
            train_step_count += 1
            run_batch_size = batch_idx * len(inputs)
            percent = 100. * batch_idx / len(train_loader)
            if batch_idx % self.log_interval == 0:
                training_writer.flush()
                print(
                    f"Epoch:{epoch} [{run_batch_size}/{len(train_loader.dataset)} ({percent:.0f}%)] {loss.item():.6f}")
        avg_training_loss = running_loss / train_step_count
        summary_writer.add_scalar("Epoch/train loss", avg_training_loss, epoch)
        self.save_training_model_state(epoch, avg_training_loss)

    def val_step(self, val_loader, epoch, training_writer, summary_writer):
        nn_model = self.nn_model
        nn_model.eval()
        running_loss = 0.0
        step_count = 0
        for batch_idx, (inputs, labels) in enumerate(val_loader):

            # output = nn_model(inputs)
            # valid_loss = nn_model.compute_loss(output, labels)
            valid_loss = self._invoke_model_validation_step(inputs, labels)

            running_loss += valid_loss
            training_writer.add_scalar("batch/validation loss", valid_loss, batch_idx)
            run_batch_size = batch_idx * len(inputs)
            step_count += 1
            percent = 100. * batch_idx / len(val_loader)
            if batch_idx % self.log_interval == 0:
                training_writer.flush()
                print(
                    f"Validation Epoch:{epoch} [{run_batch_size}/{len(val_loader.dataset)} "
                    f"({percent:.0f}%)] "
                    f"{valid_loss.item():.6f}")
        avg_training_loss = running_loss / step_count
        summary_writer.add_scalar("Epoch/validation loss", avg_training_loss, epoch)
        if self.min_valid_loss > running_loss:
            self.min_valid_loss = running_loss
            self.save_last_model_state()

    def _invoke_model_validation_step(self, x, y):
        if not hasattr(self.nn_model, 'validation_step'):
            return self.nn_model.train_step(x, y)
        return self.nn_model.validation_step(x, y)

    def _to_device(self, images, targets):
        images = [image.to(self.device) for image in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        return images, targets
