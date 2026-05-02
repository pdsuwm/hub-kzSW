# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：尝试完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。

"""

class_num = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super(TorchModel, self).__init__()

        self.net = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.BatchNorm1d(hidden_size), # 稳住数据分布
        nn.ReLU(),
        
        nn.Linear(hidden_size, hidden_size), # 增加一层
        nn.BatchNorm1d(hidden_size), # 稳住数据分布
        nn.ReLU(),
        nn.Linear(hidden_size, class_num)
        )
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y_pred = self.net(x)  # (batch_size, input_size) -> (batch_size, hidden_size)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            # return y_pred  # 输出预测结果
            # return nn.Softmax(dim=1)(y_pred)  # 输出预测结果
            return torch.softmax(y_pred, dim=1) 


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，哪一维数字最大就属于第几类
def build_sample():
    x = np.random.random(class_num)
    if random.random() < 0.5: # 50% 的概率制造极度接近的数字
        idx1, idx2 = random.sample(range(class_num), 2)
        base_val = random.random()
        x[idx1] = base_val
        x[idx2] = base_val + 0.0001 # 极其微小的差距
    label = x.argmax()
    return x, label


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # print(X)
    # print(Y)
    X = np.array(X)
    Y = np.array(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    x = x.to(device)
    y = y.to(device)
    counts = torch.bincount(y, minlength=class_num)
    for i, cnt in enumerate(counts):
        print(f"类别{i}：{cnt.item()}个")
    correct = 0
    with torch.no_grad():
        y_pred = model(x).argmax(dim=1)  # 模型预测 model.forward(x)
        correct = (y_pred == y).sum().item()
    acc = correct / test_sample_num
    print(f"正确预测个数：{correct}, 正确率：{acc}")
    return acc


def main():
    # 配置参数
    epoch_num = 30  # 训练轮数
    batch_size = 200  # 每次训练样本个数
    train_sample = 100000  # 每轮训练总共训练的样本总数
    input_size = class_num  # 输入向量维度
    # learning_rate = 0.01  # 学习率
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size).to(device)
    print(next(model.parameters()).device)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=8, gamma=0.1)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    # 创建数据加载器
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y in train_loader:
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = x.to(device)
            y = y.to(device)
            
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        scheduler.step()  # 更新学习率
        current_lr = optim.param_groups[0]['lr']
        print("当前学习率：%f" % current_lr)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = class_num
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        probs = [f"{p:.4f}" for p in res.tolist()]
        print(f"输入：{vec}, 预测类别：{res.argmax()}, 概率值：{probs}")  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)
