import numpy as np
import torch

import TorchModelTrain

"""
使用训练好的模型做预测
"""


def test_model(input_vec, model_path="diyModel.bin"):
    # 创建模型
    model = TorchModelTrain.MultiClassModel(5, 5)
    # 加载模型参数
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        predicted_class = torch.argmax(res).item()  # 获取概率最大的类别索引
        max_prob = torch.max(res).item()  # 获取最大概率值
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, predicted_class, max_prob))  # 打印结果


# 使用示例
if __name__ == "__main__":
    # 创建5个测试样本 (这里需要根据你的模型输入维度调整)
    test_samples = np.array([
        np.random.rand(5),  # 样本1
        np.random.rand(5),  # 样本2
        np.random.rand(5),  # 样本3
        np.random.rand(5),  # 样本4
        np.random.rand(5)  # 样本5
    ])

    # 测试样本
    test_model(test_samples)
