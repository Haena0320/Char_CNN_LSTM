import sys, os
sys.path.append(os.getcwd())
from src.utils import *
from src.train import *
import torch
from torch.utils.tensorboard import SummaryWriter 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="!!")
parser.add_argument("--model", type=str, default="s")
parser.add_argument("--dataset", type=str, default="PTB")

parser.add_argument("--config", type=str, default="default")
parser.add_argument("--log", type=str, default="log")
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--epochs", type=int, default=25)

parser.add_argument("--dropout_p", type=float, default=0.5)
parser.add_argument("--optim", type=str, default="sgd")
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--use_earlystop", type=int, default=1)
parser.add_argument("--use_batch_norm", type=int, default=0)

parser.add_argument("--total_steps", type=int, default=15000)
parser.add_argument("--eval_period", type=int, default=100)

args = parser.parse_args()
config = load_config(args.config)

use_cuda = torch.cuda.is_available()
device = "use_cuda {}".format(args.gpu) if use_cuda and args.gpu is not None else "cpu"


    
assert args.model in ["s", "l"]
assert args.dataset in ["PTB", "CS", "DE", "EN", "ES", "FR", "RU"]

# log file
lg = get_logger()
oj = os.path.join

args.log = "./{}/{}".format(args.log, args.model)
data_loc = oj(args.log, args.dataset)
loss_loc = oj(data_loc, "loss")
ckpnt_loc = oj(data_loc, "ckpnt")

if not os.path.exists(args.log):
    os.mkdir(args.log)
    os.mkdir(data_loc)
    os.mkdir(loss_loc)
    os.mkdir(ckpnt_loc)
    
writer = SummaryWriter(loss_loc)


from src.model import Char_CNN
import src.train as train

# data load

train_loader = train.get_data_loader(config, args.dataset, "train")
dev_loader = train.get_data_loader(config, args.dataset, "dev")
test_loader = train.get_data_loader(config, args.dataset, "test")

print("dataset iternation num : train {} | dev {} | test {}".format(len(train_loader), len(dev_loader), len(test_loader)))

# model load
model = Char_CNN(args, config)

# for name, param in model.named_parameters():
#     print(name, ":", param.requires_grad)

# train load
trainer = train.get_trainer(config, args, device, train_loader, writer, type="train")
dev_trainer = train.get_trainer(config, args, device, dev_loader, writer, type="dev")
test_trainer = train.get_trainer(config, args, device, test_loader, writer, type="test")

optimizer = train.get_optimizer(model,args.optim)
scheduler = train.get_lr_schedular(optimizer)

trainer.init_optimizer(optimizer)
trainer.init_schedular(scheduler)

early_stop_loss = []
result = list()
best_perpelxity = 10000
best_model_loc = oj(ckpnt_loc, "best_model.pkl")

for epoch in range(1, args.epochs+1):
    trainer.train_epoch(model, epoch)
    valid_loss = dev_trainer.train_epoch(model, epoch, trainer.global_step)
    early_stop_loss.append(valid_loss)
    #if args.use_earlystop and early_stop_loss[-2] < early_stop_loss[-1]:
    #    break
    if dev_trainer.perplexity < best_perpelxity:
        torch.save({"epoch":epoch,"model_stat_dict":model.state_dict()},best_model_loc)

best_model = torch.load(best_model_loc)

for epoch in range(1, args.epoch+1):
    test_trainer.train_epoch(best_model, epoch)









    

    

    
    

