# PyTorch MNIST 训练程序使用指南

## 概述
这是一个使用PyTorch训练MNIST手写数字识别模型的程序，支持多种模型架构和训练配置选项。

## 环境要求
- Python 3.x
- PyTorch 1.0+
- torchvision
- argparse (Python标准库)

## 安装依赖
```bash
pip install torch torchvision
```

## 使用方式

### 基本训练命令
```bash
python train.py
```

### 完整参数选项
```bash
python train.py \
  --batch-size 128 \               # 训练batch大小 (默认64)
  --test-batch-size 500 \          # 测试batch大小 (默认1000)
  --epochs 20 \                    # 训练轮数 (默认10)
  --learning-rate 0.001 \          # 学习率 (默认0.01)
  --gamma 0.7 \                    # 学习率衰减率 (默认0.5)
  --use-cuda False \               # 禁用GPU加速 (默认启用)
  --dry-run \                      # 快速验证单次训练流程
  --seed 42 \                      # 随机种子 (默认1)
  --log-interval 20 \              # 日志打印间隔 (默认每10个batch)
  --save-model False \             # 不保存模型 (默认保存)
  --load_state_dict "model.pth" \  # 加载预训练模型
  --model "ResNet" \               # 选择模型架构 (默认LeNet)
  --num-train-samples 30000 \      # 使用部分训练数据 (默认60000)
  --num-test-samples 5000          # 使用部分测试数据 (默认10000)
```

### 常用命令示例

1. **快速验证模型可行性**
```bash
python train.py --dry-run --epochs 1
```

2. **使用ResNet训练并保存模型**
```bash
python train.py --model ResNet --epochs 20 --save-model
```

3. **加载预训练模型继续训练**
```bash
python train.py --load_state_dict mnist_lenet.pth --epochs 5
```

4. **使用部分数据进行实验**
```bash
python train.py --num-train-samples 10000 --num-test-samples 2000
```

5. **在CPU上训练**
```bash
python train.py --use-cuda False
```

## 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--batch-size` | int | 64 | 训练batch大小 |
| `--test-batch-size` | int | 1000 | 测试batch大小 |
| `--epochs` | int | 10 | 训练迭代轮数 |
| `--learning-rate` | float | 0.01 | 初始学习率 |
| `--gamma` | float | 0.5 | 学习率衰减系数 |
| `--use-cuda` | bool | True | 是否使用GPU加速 |
| `--dry-run` | bool | False | 快速模式(仅验证单次流程) |
| `--seed` | int | 1 | 随机种子 |
| `--log-interval` | int | 10 | 训练日志打印间隔 |
| `--save-model` | bool | True | 是否保存训练好的模型 |
| `--load_state_dict` | str | "no" | 预训练模型路径 |
| `--model` | str | "LeNet" | 模型架构选择 |
| `--num-train-samples` | int | 60000 | 使用的训练样本数量 |
| `--num-test-samples` | int | 10000 | 使用的测试样本数量 |

## 模型保存
训练完成后，模型会默认保存为：
- `mnist_lenet.pth` (当使用LeNet模型时)
- `mnist_resnet.pth` (当使用ResNet模型时)

## 注意事项
1. 当CUDA可用且`--use-cuda=True`时自动使用GPU加速
2. `--dry-run`模式用于快速验证程序能否正常运行
3. 随机种子(`--seed`)确保实验可复现性
4. 学习率调度器使用StepLR，按`gamma`值衰减学习率
5. 可通过调整`--num-train-samples`模拟小样本训练场景

## 支持的模型架构
在`--model`参数中可指定：
- `LeNet`: 经典卷积神经网络
- `ResNet`: 残差网络(需代码实现支持)
- 其他自定义模型(需代码实现支持)
