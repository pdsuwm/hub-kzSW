import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

'''
基于pytorch框架编写分类训练模型
实现一个自行构造的分类任务
分类原则：一个向量，向量中最大的数在第几维，则认为是第几类
case:
[1,2,3,4,5,6,7]   属于第7类
[2,5,6,8,9,1,3]   属于第5类

'''

class TorchModel(nn.Module):
    '''模型参数'''
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.loss = nn.functional.cross_entropy

    '''前向传播'''
    def forward(self, x, y=None):
        logits = self.linear(x)
        if y is not None:
            return self.loss(logits, y.squeeze(-1))
        return logits
    
def build_sample():
    '''生成一个训练样本'''
    x = np.random.random(7)
    y = int(np.argmax(x))
    return x, y

def build_dataset(total_sample_num):
    '''生成批量的样本，打包成pytorch张量'''
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    x_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(Y, dtype=np.int64)
    return torch.from_numpy(x_arr), torch.from_numpy(y_arr)

def evaluate(model):
    '''测试每轮分类准确率'''
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    y_true = y.squeeze(-1)
    counts = [int((y_true == c).sum()) for c in range(7)]
    print("本次预测集各类样本数：%s" % counts)
    correct, wrong = 0, 0
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=-1)
        for p, t in zip(pred, y_true):
            if int(p) == int(t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d，正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    '''训练主循环'''
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 7
    num_classes = 7
    learning_rate = 0.01

    model = TorchModel(input_size, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch  + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    
    torch.save(model.state_dict(), "class_model.bin")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label='acc')
    plt.plot(range(len(log)), [l[1] for l in log], label='loss')
    plt.legend()
    plt.show()

def predict(model_path, input_vec):
    '''加载已训练模型，给定输入做预测'''
    input_size = 7
    num_classes = 7
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        logits = model.forward(torch.FloatTensor(input_vec))
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1)
    for vec, p, pr in zip(input_vec, pred, probs):
        print("输入：%s, 预测类别：%d, 各类概率： %s" % (vec, int(p), pr.numpy()))

if __name__ == "__main__":
    main()
    # test_vec = [
    #   [0.1, 0.2, 0.9, 0.3, 0.1, 0.05, 0.1],
    #   [0.8, 0.1, 0.1, 0.05, 0.02, 0.01, 0.02],
    # ]

    # predict("class_model.bin", test_vec)