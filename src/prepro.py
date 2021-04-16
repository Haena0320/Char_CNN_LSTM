import re
import numpy as np
import torch


def get_vocab(data_list, dataset):
    word2id = dict()
    char2id = dict()
    word2id["<eos>"] = 0
    char2id["<padding>"] = 0
    w_id, c_id = 1, 1
    max_word_len = 0
    for data in data_list:
        f = open(data, encoding="utf-8")
        for line in f:
            words = line_to_words(line, dataset)+["<eos>"]
            for word in words:
                max_word_len = max(len(word), max_word_len)
                if word not in word2id:
                    word2id[word] = w_id
                    w_id += 1

                for c in list(word):
                    if c not in char2id:
                        char2id[c] = c_id
                        c_id += 1
    print("max_word_length {}".format(max_word_len))
    return word2id, char2id, max_word_len


def line_to_words(line, dataset):
    clean_line = clean_str(line)
    clean_line = clean_line.split()

    return clean_line

def clean_str(string):
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\']", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " ( ", string)
    # string = re.sub(r"\)", " ) ", string)
    # string = re.sub(r"\?", " ? ", string)
    # string = re.sub(r"\s{2,}", " ", string)
    # return string.strip().lower()
    return string


def load_data(dataset, f_names, debug=0):
    #f_names = [train_path, dev_path, test_path]
    word2id, char2id, max_word_len = get_vocab(data_list=f_names, dataset=dataset)

    label = [ [] for i in range(len(f_names))]
    data =  [ [] for i in range(len(f_names))]
    files = [open(f, "r") for f in f_names]

    f_data = []
    f_label =[]
    for d,l, f in zip(data,label, files):
        for line in f:
            words = line_to_words(line, dataset) +["<eos>"]
            sent = [word2id[word] for word in words]
            l.extend(sent)
            for word in words:
                word2char = [char2id[c] for c in list(word)]
                if len(word2char) < max_word_len:
                    word2char += [0]*(max_word_len-len(word2char))
                else:
                    word2char = word2char[:max_word_len]
                d.append(word2char)

        cut = len(d)%35
        d = d[:-cut]
        l = l[1:-cut+1]
        print(len(d))
        d = np.array(d).reshape(-1, 35, max_word_len)
        l = np.array(l).reshape(-1, 35)

        f_data.append(d)
        f_label.append(l)
        print("data shape {} | label shape {}".format(d.shape, l.shape))
        assert len(d) == len(l)
        f.close()
    return word2id, char2id, f_data, f_label

