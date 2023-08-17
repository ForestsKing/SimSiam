import os
from time import time

import numpy as np
import torch
from torch.nn import CosineSimilarity
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.pretrain import DatasetPretrain
from model.simsiam import SimSiam
from utils.stop import EarlyStop


class ExpPretrain:
    def __init__(self, args, setting):
        print('\n>>>>>>>>  initing (pretrain) : {}  <<<<<<<<\n'.format(setting))

        self.args = args
        self.setting = setting

        self._acquire_device()
        self._make_dirs()
        self._get_loader()
        self._get_model()

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.devices)
            self.device = torch.device('cuda:{}'.format(self.args.devices))
            print('Use GPU: cuda:{}'.format(self.args.devices))
        else:
            self.device = torch.device('cpu')
            print('Use CPU')

    def _make_dirs(self):
        self.data_path = self.args.data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        self.model_path = self.args.save_path + '/' + self.setting + '/pretrain/model/'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.log_path = self.args.save_path + '/' + self.setting + '/pretrain/log/'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def _get_loader(self):
        train_set = DatasetPretrain(self.data_path, train=True, download=self.args.download)
        test_set = DatasetPretrain(self.data_path, train=False, download=self.args.download)

        self.train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False)

    def _get_model(self):
        self.model = SimSiam(self.args).to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=self.args.lr)
        self.early_stop = EarlyStop(patience=self.args.patience, path=self.model_path)
        self.loss_fn = CosineSimilarity()
        self.writer = SummaryWriter(log_dir=self.log_path)

        torch.save(self.model.state_dict(), self.model_path + '/' + 'checkpoint.pth')

    def train(self):
        print('\n>>>>>>>>  training (pretrain) : {}  <<<<<<<<\n'.format(self.setting))

        for e in range(self.args.epoch):
            start = time()
            train_loss, valid_loss = [], []

            self.model.train()
            for (batch_x1, batch_x2) in self.train_loader:
                self.optimizer.zero_grad()

                batch_x1 = batch_x1.to(self.device)
                batch_x2 = batch_x2.to(self.device)
                z1, z2, p1, p2 = self.model(batch_x1, batch_x2)

                if self.args.stop_grad:
                    z1, z2 = z1.detach(), z2.detach()
                loss = 0.5 * self.loss_fn(p1, z2).mean() + 0.5 * self.loss_fn(p2, z1).mean()

                train_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                for (batch_x1, batch_x2) in self.test_loader:
                    batch_x1 = batch_x1.to(self.device)
                    batch_x2 = batch_x2.to(self.device)
                    z1, z2, p1, p2 = self.model(batch_x1, batch_x2)

                    if self.args.stop_grad:
                        z1, z2 = z1.detach(), z2.detach()
                    loss = 0.5 * self.loss_fn(p1, z2).mean() + 0.5 * self.loss_fn(p2, z1).mean()

                    valid_loss.append(loss.item())

            train_loss, valid_loss = np.mean(train_loss), np.mean(valid_loss)
            end = time()

            print("Epoch: {0} || Train Loss: {1:.6f} Valid Loss: {2:.6f}|| Cost: {3:.6f}".format(
                e, train_loss, valid_loss, end - start)
            )

            self.writer.add_scalar('train/train_loss', train_loss, e)
            self.writer.add_scalar('train/valid_loss', valid_loss, e)

            self.early_stop(valid_loss, self.model)
            if self.early_stop.early_stop:
                break

        self.writer.close()

    def get_encoder(self):
        self.model.load_state_dict(torch.load(self.model_path + '/' + 'checkpoint.pth'))
        return self.model.encoder
