import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from loadDataset import cost_Time
from bert_SourceCode.optimization import BertAdam


# 模型训练
def train(config, model, train_iter, dev_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    network_param = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in network_param)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in network_param)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    current_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    no_improve = False
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (contents, labels) in enumerate(train_iter):
            outputs = model(contents)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if current_batch % 100 == 0:
                real_label = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(real_label, predict)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_model_path)
                    improve = '*'
                    last_improve = current_batch
                else:
                    improve = ''
                costTime = cost_Time(start_time)
                msg = 'Iter: {0:>6}, Train_Loss: {1:>5.2}, Train_Acc: {2:>6.2%}, Dev_Loss: {3:>5.2}, Dev_Acc: {4:>6.2%}, Time: {5} {6}'
                print(msg.format(current_batch, loss.item(), train_acc, dev_loss, dev_acc, costTime, improve))
                model.train()
            current_batch += 1
            if current_batch - last_improve > config.require_improve:
                print("No optimization for a long time, auto-stopping...")
                no_improve = True
                break
        if no_improve:
            break


# 模型评估
def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    labels_all = np.array([], dtype=int)
    predicts_all = np.array([], dtype=int)
    with torch.no_grad():
        for contents, labels in data_iter:
            outputs = model(contents)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predicts = torch.max(outputs, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predicts_all = np.append(predicts_all, predicts)
    acc = metrics.accuracy_score(labels_all, predicts_all)
    avg_loss = loss_total / len(data_iter)
    if test:
        report = metrics.classification_report(labels_all, predicts_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predicts_all)
        return acc, avg_loss, report, confusion
    return acc, avg_loss


# 模型测试
def test(config, model, data_iter):
    model.load_state_dict(torch.load(config.save_model_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, report, confusion = evaluate(config, model, data_iter, test=True)
    msg = 'Test_Loss: {0:>5.2}, Test_Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score ...")
    print(report)
    print("Confusion Matrix ...")
    print(confusion)
    print("Cost Time:", cost_Time(start_time))
















