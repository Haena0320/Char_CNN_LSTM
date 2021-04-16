import sys, os
sys.path.append(os.getcwd())
from src.utils import *
from src.prepro import *
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default=">_<")
parser.add_argument("--model", type=str, default="small")
parser.add_argument("--dataset",type=str, default="PTB" ) ## dataset
parser.add_argument("--data_size", type=str, default="s") ## small size dataset
parser.add_argument("--config", type=str, default="default")
parser.add_argument("--debug", type=int, default=0)

args = parser.parse_args()
config = load_config(args.config)
oj = os.path.join

assert args.model in ["small", "large"]
assert args.dataset in ["PTB", "CS", "DE", "EN", "ES", "FR", "RU"]
assert args.data_size in ["s", "l"]

## preprocessing code
#dataset =["PTB", "CS", "DE", "EN", "ES", "FR", "RU"]
dataset = ["FR"]
for data in dataset:
    prepro = dict()
    files = config.raw_data[data]
    s = oj(config.preprocessed, data)
    if not os.path.exists(s):
        os.mkdir(s)

    print("processsing : {}".format(data))
    word2id, char2id, prepro_data, label = load_data(data, files, args.debug)
    print(prepro_data[0].shape)
    print(label[0].shape)
    prepro["train"] = prepro_data[0]
    prepro["train_label"] = label[0]
    prepro["dev"] = prepro_data[1]
    prepro["dev_label"] = label[1]
    prepro["test"] = prepro_data[2]
    prepro["test_label"] = label[2]

    if args.debug:
        print("word2id {} embeddings".format(len(word2id.items())))
        print("char2id {} embeddings".format(len(char2id.items())))
        print("char2id {}".format(char2id.keys()))

    torch.save(word2id, s+"/"+"word2id.pkl")
    torch.save(char2id, s+"/"+"char2id.pkl")
    torch.save(prepro, s+"/"+"prepro.pkl")

print("finished !! ")


