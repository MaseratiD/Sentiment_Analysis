import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_SourceCode.tokenization import BertTokenizer
from bert_SourceCode.modeling import BertModel


class Config(object):
    def __init__(self, dataDir):
        self.model_name = 'bert'  # 模型名称
        self.train_data = dataDir + '/train.txt'  # 训练集
        self.dev_data = dataDir + '/dev.txt'  # 验证集
        self.text_data = dataDir + '/test.txt'  # 测试集
        self.class_list = [x.strip() for x in open(dataDir + '/class.txt').readlines()]  # label类别（几分类）
        self.save_model_path = 'trainedModel/' + self.model_name + '.pkl'  # 存储训练好的模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备使用

        self.num_classes = len(self.class_list)  # 分类数
        self.pad_size = 32  # 每句话处理的长度（短了补长了切）
        self.batch_size = 128  # mini-batch的大小
        self.hidden_size = 768  # 隐层大小
        self.learning_rate = 5e-5  # 学习率
        self.num_epochs = 3  # epoch数
        self.require_improve = 1000  # 若超过1000 batch仍无提升，则提前结束

        self.bert_SourceCode_path = './bert_SourceCode'  # bert源码
        self.pretrainedModel_path = './bert_pretrainedModel'  # 下载的bert预训练模型
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrainedModel_path)  # BertTokenizer

        self.unit_sizes = (2, 3, 4)
        self.num_units = 256
        self.dropout = 0.1


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 从bert源码中加载模型
        self.bert = BertModel.from_pretrained(config.pretrainedModel_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList([nn.Conv2d(1, config.num_units, (k, config.hidden_size)) for k in config.unit_sizes])
        self.dropout = nn.Dropout(config.dropout)

        self.fc_cnn = nn.Linear(config.num_units * len(config.unit_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        content = x[0]
        label = x[1]
        mask = x[2]
        encoder_out, text_cls = self.bert(content, attention_mask=mask, output_all_encoded_layers=False)
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out
