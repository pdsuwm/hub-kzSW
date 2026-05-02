import torch
from torch import nn

"""
完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类
"""

# 样本大小
MAX_SAMPLES = 1000
# 特征值数量
FEATURE_NUM = 3

# 生成数据
def generate_data(max_samples = MAX_SAMPLES, feature_num = FEATURE_NUM):
    x = torch.randn(max_samples, feature_num)
    y_true = x.argmax(dim=1)
    return x, y_true


# 模型定义
class MultiClassModule(nn.Module):
    """
    多分类任务
    输入：特征值
    输出：预测值
    """

    def __init__(self, feature_num = FEATURE_NUM, layer1_num = 5, layer2_num = 5):
        super().__init__()
        self.layer1 = nn.Linear(feature_num, layer1_num)
        self.layer2 = nn.Linear(layer1_num, layer2_num)
        self.out = nn.Linear(layer2_num, feature_num)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.out(x)
        return x

# 训练
def train(module, x, y_true):
    # 1.定义损失函数，内部有softmax
    loss_fn = nn.CrossEntropyLoss()

    # 2.定义优化器
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)

    # 3.开始训练
    module.train()

    epochs = 100
    batch_size = 10

    for epoch in range(epochs):
        for i in range(0, x.shape[0], batch_size):
            # 获取数据
            x_batch = x[i:i+batch_size]
            y_true_batch = y_true[i:i+batch_size]

            # 模型预测
            y_pred = module(x_batch)

            # 计算损失
            loss = loss_fn(y_pred, y_true_batch)

            # 反向传播
            loss.backward()

            # 优化参数
            optimizer.step()

            # 梯度清零
            optimizer.zero_grad()

    print(f'第{epoch}轮训练结束，损失值为{loss.item()}')

if __name__ == '__main__':

    # 1.数据准备
    x_test, y_true_test = generate_data()

    # 2.模型定义
    module = MultiClassModule()

    # 3.模型训练
    print('开始训练...')
    train(module, x_test, y_true_test)

    # 4.模型测试
    print('开始测试...')
    module.eval()
    with torch.no_grad():
        x_test, y_true_test = generate_data(1)
        y_pred_logits = module(x_test)

        # 转换为概率
        y_pred_probs = torch.softmax(y_pred_logits, dim=-1)
        # 转换为类别
        y_pred_class = y_pred_logits.argmax(dim=-1)

        print('测试输入：', x_test.tolist())
        print('真实类别：', y_true_test.item())
        print('预测概率：', y_pred_probs.tolist())
        print('预测类别：', y_pred_class.item())






