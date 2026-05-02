import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个多分类任务的训练
规律：x是一个5维向量，哪一维数字最大就属于第几类

"""


class MultiClassModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)  # 线性层
        # self.activation = nn.Softmax(dim=1)  # softmax激活函数
        self.criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 前向传播
    def forward(self, x):
        y_pred = self.linear(x)
        # y_pred = self.activation(x)
        return y_pred

    # 损失函数
    def loss(self, y_pred, y):
        return self.criterion(y_pred, y)


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，哪一维数字最大就属于第几类
def generate_sample():
    x = np.random.rand(5)
    y = np.argmax(x)
    return x, y


# 使用generate_sample方法随机生成一批样本
# 每个类别均匀生成样本数
def generate_batch(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = generate_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))


# 测试代码
# 用来测试每轮模型的准确率
def test_model(model):
    model.eval()
    test_sample_num = 100
    x, y = generate_batch(test_sample_num)
    y_pred = model(x)
    # 打印输入样本
    # print(f"输入样本:{x}")
    # 打印真实标签
    # print(f"真实标签:{y}")
    # 打印预测结果
    # print(f"预测结果:{y_pred}")
    # 打印预测标签
    # print(f"预测标签:{y_pred.argmax(dim=1)}")
    # 计算准确率
    accuracy = (y_pred.argmax(dim=1) == y).float().mean()
    # print("测试准确率:%f" % accuracy)
    return accuracy


def main():
    # 配置参数
    epoch_num = 30  # 训练轮数
    batch_size = 200  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    output_size = 5  # 输出向量维度

    # 创建模型
    model = MultiClassModel(input_size, output_size)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 创建训练集，正常任务是读取训练集
    train_x, train_y = generate_batch(train_sample)
    # 记录每轮模型的准确率和loss
    log = []
    # 训练模型
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            y_pred = model(x)  # 前向传播
            loss = model.loss(y_pred, y)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            optimizer.zero_grad()  # 清空梯度
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        # 测试模型
        accuracy = test_model(model)
        print(f"第{epoch + 1}轮测试准确率:{accuracy}")
        log.append([accuracy, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "diyModel.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    main()
