# coding:utf8

# 解决 OpenMP 库冲突问题
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，哪一维数字最大就属于第几类（类别：0,1,2,3,4）

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层，输出num_classes个logits
        self.activation = torch.softmax  # softmax归一化函数，输出概率分布
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失（适合多分类）

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        y_pred = self.activation(x, dim=-1)  # (batch_size, num_classes) -> (batch_size, num_classes)，概率分布
        if y is not None:
            # y的形状需要是(batch_size,)，cross_entropy要求标签是类别索引
            y = y.squeeze().long()  # 确保y是1维的整数张量
            return self.loss(x, y)  # 注意：cross_entropy输入logits，不需要softmax
        else:
            return y_pred  # 输出预测结果（概率分布）


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，找到最大值的索引作为类别（0-4）
def build_sample():
    x = np.random.random(5)
    # 找出最大值的索引
    class_idx = np.argmax(x)
    return x, class_idx


# 随机生成一批样本
# 各类别均匀生成（随机数据下各类别概率大致相等）
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 保持为二维，后面会squeeze
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)


    # 统计每个类别的样本数
    unique, counts = np.unique(y.numpy(), return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("本次预测集样本分布：", class_counts)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，得到概率分布 (batch_size, num_classes)
        y_pred_class = torch.argmax(y_pred, dim=1)  # 取概率最大的类别作为预测类别

        correct = (y_pred_class == y).sum().item()
        wrong = len(y) - correct

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数量（5维，对应5个类别）
    learning_rate = 0.01  # 学习率

    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model_multiclass.bin")

    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测，得到概率分布

    for vec, res in zip(input_vec, result):
        pred_class = torch.argmax(res).item()
        print("输入：%s" % vec)
        print("预测类别：%d" % pred_class)
        print("各类别概率：", [f"{p:.4f}" for p in res.tolist()])
        print("-" * 50)


if __name__ == "__main__":
    main()

    # 测试预测
    test_vec = [
        [0.1, 0.2, 0.3, 0.4, 0.5],  # 第4类（索引4）最大
        [0.9, 0.1, 0.1, 0.1, 0.1],  # 第0类最大
        [0.2, 0.8, 0.1, 0.1, 0.1],  # 第1类最大
        [0.1, 0.2, 0.7, 0.1, 0.1],  # 第2类最大
        [0.1, 0.1, 0.1, 0.6, 0.1],  # 第3类最大
    ]
    predict("model_multiclass.bin", test_vec)
