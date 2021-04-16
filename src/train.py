import torch
from torch import flatten
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

def get_data_loader(config=None, dataset=None, data_type="train", shuffle=False, workers=10, drop_last=True):
    bs = config.train.bs
    dataset = Make_Dataset(config.path_preprocessed[dataset]["prepro"], data_type)
    data_loader = DataLoader(dataset, batch_size = bs, shuffle=shuffle, num_workers=workers, drop_last=drop_last)
    return data_loader

class Make_Dataset(Dataset):
    def __init__(self, file_path, data_type):
        self.dataset = torch.load(file_path)

        if data_type == "train":
            self.data = np.array(self.dataset["train"])
            print(self.data.shape)
            self.data_label = self.dataset["train_label"]
            assert len(self.data) == len(self.data_label)

        elif data_type == "dev":
            self.data = np.array(self.dataset["dev"])
            self.data_label = self.dataset["dev_label"]
            assert len(self.data) == len(self.data_label)

        else:
            self.data = np.array(self.dataset["test"])
            self.data_label = self.dataset["test_label"]
            assert len(self.data) == len(self.data_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): # idx = batch_size * 35 (back_propre)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ret_dict = dict()
        ret_dict["data"] = self.data[idx]
        ret_dict["data_label"] = self.data_label[idx]

        return ret_dict


def get_trainer(config, args, device, data_loader, log_writer, type):
    return Trainer(config, args, device, data_loader, log_writer, type)

def get_optimizer(model, args_optim):
    if args_optim == "sgd":
        return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def get_lr_schedular(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1)

class Trainer:
    def __init__(self, config, args, device, data_loader, writer, type):
        self.config = config
        self.args = args
        self.device = device
        self.data_loader = data_loader
        self.writer = writer
        self.type = type
        self.bs = config.train.bs
        self.bptt = config.train.bptt

        self.loss_function = CrossEntropyLoss()
        self.global_step = 0

    def init_optimizer(self, optimizer):
        self.optimizer = optimizer

    def init_schedular(self, schedular):
        self.schedular = schedular

    def train_epoch(self, model, epoch, global_step=None):
        if self.type=="train":
            model.train()
        else:
            model.eval()

        model.to(self.device)
        loss_save = list()

        for data in tqdm(self.data_loader):
            x = data["data"].to(self.device)
            label = data["data_label"].to(self.device)

            y = model.forward(x)
            label = torch.flatten(label, start_dim=0, end_dim=1)
            loss = self.loss_function(y, label)

            if self.type =="train":
                self.optim_process(model, loss)
                self.global_step +=1
                self.write_log(loss, self.global_step)

            else:
                loss_save.append(loss)
                self.perplexity = torch.exp(loss)
                print("mode {} | perplexity {}".format(self.type, self.perplexity))

    def optim_process(self, model, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.train.clip)
        self.optimizer.step()
        self.schedular.step()

    def write_log(self, loss, global_step):
        if self.type == "train":
            lr = self.optimizer.param_groups[0]["lr"]
            self.writer.add_scalar("train/loss", loss, global_step)
            self.writer.add_scalar("lr/loss", lr, global_step)

        else:
            self.writer.add_scalar("valid/loss", loss, global_step)














