import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze())  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果

def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 预估
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        pred_class = torch.argmax(y_pred, dim=1)
        for p, t in zip(pred_class, y):
            if p == t:
                correct += 1
            else:
                wrong += 1
    print(f"正确预测: {correct}, 错误预测: {wrong}, 准确率: {correct / (correct + wrong):.2f}")
    return correct / (correct + wrong)

def main():
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    total_epochs = 20
    batch_size = 20
    trans_sample = 5000
    trans_x, trans_y = build_dataset(trans_sample)
    log = []
    for epoch in range(total_epochs):
        model.train()
        watch_loss = []
        for i in range(trans_sample // batch_size):
            batch_x = trans_x[i * batch_size:(i + 1) * batch_size]
            batch_y = trans_y[i * batch_size:(i + 1) * batch_size]
            loss = model(batch_x, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        # 打印本轮结果
        avg_loss = np.mean(watch_loss)
        print(f"Epoch [{epoch + 1}/{total_epochs}], Loss: {avg_loss.item():.4f}")
        acc = evaluate(model)
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

def predict(model_path, test_input):
    model = TorchModel(input_size=5, num_classes=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(test_input)
        y_pred = model(x)
        pred_class = torch.argmax(y_pred, dim=1)
        print("\n预测结果:")
        for i, (vec, pred) in enumerate(zip(test_input, pred_class)):
            print(f"输入: {vec}, 预测类别: {pred.item()}")

if __name__ == "__main__":
    main()

    test_input = [[0.1, 0.2, 0.3, 0.4, 0.5],  # 类别4
                  [0.5, 0.4, 0.3, 0.2, 0.1],  # 类别0
                  [0.2, 0.3, 0.4, 0.5, 0.1],  # 类别3
                  [0.3, 0.4, 0.5, 0.1, 0.2],  # 类别2
                  [0.4, 0.5, 0.1, 0.2, 0.3]]  # 类别1
    predict("model.bin", test_input)