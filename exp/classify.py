import os
from time import time

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.classify import DatasetClassify
from exp.pretrain import ExpPretrain
from model.classifier import Classifier
from utils.stop import EarlyStop


class ExpClassify:
    def __init__(self, args, setting):
        self.args = args
        self.setting = setting

        if self.args.encoder:
            exp = ExpPretrain(self.args, self.setting)

            if self.args.pretrain:
                exp.train()

            encoder = exp.get_encoder()

        else:
            encoder = None

        print('\n>>>>>>>>  initing (classify) : {}  <<<<<<<<\n'.format(setting))

        self._acquire_device()
        self._make_dirs()
        self._get_loader()
        self._get_model(encoder)

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

        self.model_path = self.args.save_path + '/' + self.setting + '/classify/model/'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.log_path = self.args.save_path + '/' + self.setting + '/classify/log/'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def _get_loader(self):
        train_set = DatasetClassify(self.data_path, True, self.args.download, self.args.train_ratio)
        test_set = DatasetClassify(self.data_path, False, self.args.download, self.args.train_ratio)

        self.train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False)

    def _get_model(self, encoder):
        self.model = Classifier(self.args, encoder).to(self.device)
        self.optimizer = SGD(self.model.parameters(), lr=self.args.lr)
        self.early_stop = EarlyStop(patience=self.args.patience, path=self.model_path)
        self.loss_fn = CrossEntropyLoss()
        self.writer = SummaryWriter(log_dir=self.log_path)

    def train(self):
        print('\n>>>>>>>>  training (classify) : {}  <<<<<<<<\n'.format(self.setting))

        for e in range(self.args.epoch):
            start = time()

            self.model.train()
            train_loss = []
            for (batch_x, batch_y) in self.train_loader:
                self.optimizer.zero_grad()

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                output = self.model(batch_x)
                loss = self.loss_fn(output, batch_y)

                train_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()

            train_loss = np.mean(train_loss)
            test_loss, test_acc = self.test(verbose=False, load_model=False)
            end = time()

            print("Epoch: {0} || Train Loss: {1:.6f} Test Loss: {2:.6f} Test ACC: {3:.4f} || Cost: {4:.6f}".format(
                e, train_loss, test_loss, test_acc, end - start)
            )

            self.writer.add_scalar('train/train_loss', train_loss, e)
            self.writer.add_scalar('train/test_loss', test_loss, e)
            self.writer.add_scalar('train/test_acc', test_acc, e)

            self.early_stop(test_loss, self.model)
            if self.early_stop.early_stop:
                break

        self.writer.close()

    def test(self, verbose=True, load_model=True):
        if verbose:
            print('\n>>>>>>>>  testing (classify) : {}  <<<<<<<<\n'.format(self.setting))

        if load_model:
            self.model.load_state_dict(torch.load(self.model_path + '/' + 'checkpoint.pth'))

        with torch.no_grad():
            self.model.eval()

            test_loss, test_label, test_pred = [], [], []
            for (batch_x, batch_y) in self.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                output = self.model(batch_x)
                loss = self.loss_fn(output, batch_y)

                test_loss.append(loss.item())
                test_label.append(batch_y.cpu().numpy())
                test_pred.append(output.cpu().numpy())

        test_loss = np.mean(test_loss)
        test_label = np.concatenate(test_label, axis=0)
        test_pred = np.concatenate(test_pred, axis=0)
        test_pred = np.argmax(test_pred, axis=1)

        test_acc = np.sum(test_pred == test_label) / len(test_label)

        if verbose:
            print("Test Loss: {0:.6f} || Test Acc: {1:.4f}".format(test_loss, test_acc))

        return test_loss, test_acc
