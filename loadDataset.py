# coding: UTF-8
import torch
import time
from tqdm import tqdm
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


# 加载数据
def loadDataset(config):
    print("Loading data...")
    start_Time = time.time()
    # 数据转换为[token, label, seq_len, mask]格式
    train_data = loadData(config, config.train_data, config.pad_size)
    dev_data = loadData(config, config.dev_data, config.pad_size)
    test_data = loadData(config, config.text_data, config.pad_size)

    # 转换为迭代器
    train_iter = list2iter(train_data, config)
    dev_iter = list2iter(dev_data, config)
    test_iter = list2iter(test_data, config)
    print("Cost Time: ", cost_Time(start_Time))
    return train_iter, dev_iter, test_iter


# 初始数据转换为token，生成mask
def loadData(config, dataPath, pad_size=32):
    contents = []
    with open(dataPath, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            content, label = line.split('\t')
            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_length = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            if pad_size:
                if seq_length < pad_size:
                    mask = [1] * seq_length + [0] * (pad_size - seq_length)
                    token_ids += [0] * (pad_size - seq_length)
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_length = pad_size
            contents.append((token_ids, int(label), seq_length, mask))
    return contents


class DatasetIterator(object):
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = len(dataset) // batch_size
        self.device = device
        self.residue = False
        self.index = 0
        if len(dataset) % batch_size != 0:
            self.residue = True

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

    def __iter__(self):
        return self

    def to_tensor(self, datas):
        x = torch.LongTensor([data[0] for data in datas]).to(self.device)  # content
        y = torch.LongTensor([data[1] for data in datas]).to(self.device)  # label
        seq_len = torch.LongTensor([data[2] for data in datas]).to(self.device)  # content length
        mask = torch.LongTensor([data[3] for data in datas]).to(self.device)  # content mask
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            dataset = self.dataset[self.index * self.batch_size: len(self.dataset)]
            self.index += 1
            dataset = self.to_tensor(dataset)
            return dataset
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            dataset = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            dataset = self.to_tensor(dataset)
            return dataset


# 转换为迭代器
def list2iter(dataset, config):
    return DatasetIterator(dataset, config.batch_size, config.device)


# 运行耗时
def cost_Time(start_time):
    end_time = time.time()
    return timedelta(seconds=int(round(end_time - start_time)))
