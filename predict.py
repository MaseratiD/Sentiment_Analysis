# coding: UTF-8
import time
import torch
import argparse
from importlib import import_module
from loadDataset import loadDataset, cost_Time, list2iter

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a models: Bert, BertCNN, ERNIE')
parser.add_argument('--predict', type=str, required=True, help='input the predict sentence')
args = parser.parse_args()
PAD, CLS = '[PAD]', '[CLS]'


def evaluate(model, content):
    model.eval()
    with torch.no_grad():
      for text, label in content:
        outputs = model(text)
        predict = torch.max(outputs.data, 1)[1].cpu().numpy()
    print("Predict List: ", outputs)
    print("Predict Label:", predict)


def loadData(config, sentence, pad_size=32):
    contents = []
    line = sentence.strip()
    if not line:
        return
    token = config.tokenizer.tokenize(line)
    token = [CLS] + token
    seq_len = len(token)
    mask = []
    token_ids = config.tokenizer.convert_tokens_to_ids(token)
    if pad_size:
        if len(token) < pad_size:
            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
            token_ids += ([0] * (pad_size - len(token)))
        else:
            mask = [1] * pad_size
            token_ids = token_ids[:pad_size]
            seq_len = pad_size
    contents.append((token_ids, int(-1), seq_len, mask))
    return contents


if __name__ == '__main__':
    # 根据执行命令获取要运行的模型，并引入对应的module，使用模型对应的配置参数
    model_name = args.model
    sentence = args.predict
    model_path = import_module('models.' + model_name)
    config = model_path.Config('dataset')

    # 使用训练好的模型
    model = model_path.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_model_path), False)

    # 预测
    start_time = time.time()
    contentList = loadData(config, sentence)
    contentIter = list2iter(contentList, config)
    evaluate(model, contentIter)
    print("Cost Time: ", cost_Time(start_time))


