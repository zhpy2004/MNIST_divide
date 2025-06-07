from __future__ import print_function # 这个是python当中让print都以python3的形式进行print，即把print视为函数
import argparse
import os
#import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import time
from torch.utils.data import Subset


# import network
from model.network.LeNet import LeNet
from model.network.DefaultNet import DefaultNet


graph_loss = []
graph_acc = []

def train(args, model, device, train_loader, test_loader, optimizer, scheduler):
    model.train()
    best_epoch = 0
    best_acc = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            tmp_time = time.time()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()   
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % args.log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}\t Cost time: {:.6f}s".format(
                    epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), time.time() - tmp_time
                ))
                graph_loss.append(loss.item())
                if args.dry_run:
                    break
        scheduler.step()
        end_time = time.time()
        print("Epoch {} cost {} s".format(epoch + 1, end_time - start_time))
        acc = test(model, device, test_loader)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            if args.save_model:
                torch.save(model.state_dict(), "./model/weights/{}.pt".format(args.model))
    print("Best epoch: {} | Best acc: {}".format(best_epoch, best_acc))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item() 
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()#

    test_loss /= len(test_loader.dataset)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        float(100. * correct / len(test_loader.dataset))
    ))

    graph_acc.append(100. * correct / len(test_loader.dataset))

    return float(100. * correct / len(test_loader.dataset))

# action 和 gamma , metavar的作用
def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Pytorch MNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",          # 这里的batch-size是训练的batch大小，默认是64   
                        help="input batch size for training (default : 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",   # 这里的test-batch-size是测试的batch大小，默认是1000
                        help="input batch size for testing (default : 1000)")
    parser.add_argument("--epochs", type=int, default=10, metavar="N",              # 这里的epochs是训练的轮数，默认是10
                        help="number of epochs to train (default : 64)")
    parser.add_argument("--learning-rate", type=float, default=0.01, metavar="LR",  # 这里的learning-rate是学习率，默认是0.01    
                        help="the learning rate (default : 0.1)")
    parser.add_argument("--gamma", type=float, default=0.5, metavar="M",            # 这里的gamma是学习率衰减率，默认是0.5
                        help="Learning rate step gamma (default : 0.5)")
    parser.add_argument("--use-cuda", action="store_true", default=True,            # 这里的use-cuda是是否使用cuda训练，默认是True
                        help="Using CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False,            # 这里的dry-run是是否进行快速测试，默认是False
                        help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S",                 # 这里的seed是随机种子，默认是1
                        help="random seed (default : 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",        # 这里的log-interval是训练多少个batch打印一次日志，默认是10
                        help="how many batches to wait before logging training status")
    parser.add_argument("--save-model", action = "store_true", default=True,        # 这里的save-model是是否保存模型，默认是True
                        help="For saving the current Model")
    parser.add_argument("--load_state_dict", type=str, default="no",                # 这里的load_state_dict是是否加载模型，默认是no
                        help="load the trained model weights or not (default: no)")
    parser.add_argument("--model", type=str, default="LeNet",                       # 这里的model是选择的模型，默认是LeNet
                        help="choose the model to train (default: LeNet)")
    parser.add_argument("--num-train-samples", type=int, default=60000, metavar="N", # ** 这里的num-train-samples是训练集的数量，默认是60000
                    help="number of training samples to use (default: 60000)")
    parser.add_argument("--num-test-samples", type=int, default=10000, metavar="N", # ** 这里的num-test-samples是测试集的数量，默认是10000
                    help="number of testing samples to use (default: 10000)")

    args = parser.parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available() # not > and > or
    print("Using cuda is: {}".format(use_cuda))
    torch.manual_seed(args.seed)   

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    dataset1 = datasets.MNIST("./data", train=True, download=True,
                            transform=transform)
    dataset2 = datasets.MNIST("./data", train=False,
                            transform=transform)
    
    '''
    这里的Subset是从dataset1中随机抽取num_train_samples个样本，作为训练集
    '''
    if args.num_train_samples < len(dataset1):
        dataset1 = Subset(dataset1, range(args.num_train_samples))
    if args.num_test_samples < len(dataset2):
        dataset2 = Subset(dataset2, range(args.num_test_samples))

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model_name = args.model.lower()
    if model_name == "lenet":
        model = LeNet().to(device)
    elif model_name == "defaultnet":
        model = DefaultNet().to(device)


    #model = Net().to(device)
    model_path = Path("./model/weights/{}.pt".format(model_name))
    if model_path.exists() and args.load_state_dict == "yes":
        model.load_state_dict(torch.load(model_path))
        print("Load the last trained model.")
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9,0.999), weight_decay=0)    # 使用adam优化器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.00001)                  # 使用余弦退火调节机制

    
    train(args, model, device, train_loader, test_loader, optimizer, scheduler)
   
    create_loss_txt_path = "./model/result/{}_loss.txt".format(model_name)
    create_acc_txt_path = "./model/result/{}_acc.txt".format(model_name)
    f = open(create_loss_txt_path, "w+")
    for loss in graph_loss: 
        f.writelines("{}\n".format(loss))
    f.close()
    f = open(create_acc_txt_path, "w+")
    for acc in graph_acc:
        f.writelines("{}\n".format(acc))
    f.close()


if __name__ == "__main__":
    main()