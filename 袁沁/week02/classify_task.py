# coding:utf8
import os
import torch
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt
"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
五维判断：x是一个5维向量，向量中哪个标量最大就输出哪一维下标（0-4）

多分类：输出概率分布，使用交叉熵损失函数
"""
class MultiClassificationModel(nn.Module):
    def __init__(self, input_size, num_classes=5):
        super(MultiClassificationModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层，输出num_classes个类别
        self.loss = nn.CrossEntropyLoss()  # 使用CrossEntropyLoss模块


    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=1)  # 输出预测结果（概率分布）

# 随机生成一个5维向量，根据每个向量中最大的标量同一下标构建Y
def build_sample(dim=5):
    x = np.random.random(dim)
    # 获取最大值的索引（如果有多个最大值，取第一个）
    max_index = np.argmax(x)
    return x, max_index


# 随机生成一批样本
def build_dataset(total_sample_num, dim=5):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample(dim)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码，用来测试每轮模型的准确率
def evaluate(model, dim=5):
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num, dim)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，得到概率分布
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1

    accuracy = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


def main():
    # 配置参数
    epoch_num = 30  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 8000  # 每轮训练总共训练的样本总数
    
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数量
    learning_rate = 0.001  # 学习率
    
    # 建立模型
    model = MultiClassificationModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    
    # 创建训练集
    train_x, train_y = build_dataset(train_sample, input_size)
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        shuffle_indices = torch.randperm(train_sample)
        train_x = train_x[shuffle_indices]
        train_y = train_y[shuffle_indices]
        
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, input_size)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
  
    torch.save(model.state_dict(), "multi_classification_model.pt")
  
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = MultiClassificationModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  
    print("加载模型权重成功！")

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测，得到概率分布
    
    print("\n预测结果：")
    for vec, res in zip(input_vec, result):
        predicted_class = torch.argmax(res).item()
        max_prob = torch.max(res).item()
        print("输入：%s" % vec)
        print("预测类别：%d (概率值：%.4f)" % (predicted_class, max_prob))
        print("各类别概率分布：%s\n" % res.numpy())


if __name__ == "__main__":
    # 训练模型
    main()
    
    # 测试预测功能
    print("\n" + "="*50)
    print("测试预测功能：")
    test_vec = [
        [0.1, 0.8, 0.2, 0.3, 0.4],  # 第2维(索引1)最大
        [0.9, 0.1, 0.1, 0.1, 0.1],  # 第1维(索引0)最大
        [0.2, 0.2, 0.2, 0.7, 0.1],  # 第4维(索引3)最大
        [0.3, 0.3, 0.8, 0.2, 0.1],  # 第3维(索引2)最大
        [0.4, 0.3, 0.2, 0.1, 0.9],  # 第5维(索引4)最大
    ]
    predict("multi_classification_model.pt", test_vec)
