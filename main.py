# coding: UTF-8
import torch
import numpy as np
import argparse
from importlib import import_module
from loadDataset import loadDataset
from train_eval import train, test

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a models: Bert, BertCNN, ERNIE')
args = parser.parse_args()

if __name__ == '__main__':
    # 数据集所在目录
    dataDir = 'dataset'
    # 根据执行命令获取要运行的模型，并引入对应的module，使用模型对应的配置参数
    model_name = args.model
    model_path = import_module('models.' + model_name)
    config = model_path.Config(dataDir)

    # 保证每次的结果相同
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    # 构造初始模型
    model = model_path.Model(config).to(config.device)

    # 数据加载与处理
    train_iter, dev_iter, test_iter = loadDataset(config)
    # 训练
    train(config, model, train_iter, dev_iter)

    # 测试
    test(config, model, test_iter)
