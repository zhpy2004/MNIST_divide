from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import time
from model.network.LeNet import LeNet

from model.network.DefaultNet import DefaultNet

from torch.utils.data import Subset

list_pred = list()
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
            list_pred.append(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print("Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))

    #print(list_pred)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Pytorch MNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default : 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                        help="input batch size for testing (default : 1000)")
    parser.add_argument("--epochs", type=int, default=14, metavar="N",
                        help="number of epochs to train (default : 14)")
    parser.add_argument("--learning-rate", type=float, default=0.5, metavar="LR",
                        help="number of epochs to train (default : 14)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M",
                        help="Learning rate step gamma (default : 0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default : 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--save-model", action = "store_true", default=True,
                        help="For saving the current Model")
    parser.add_argument("--load_state_dict", type=str, default="no",
                        help="load the trained model weights or not (default: no)")
    parser.add_argument("--model", type=str, default="LeNet",
                        help="choose the model to train (default: LeNet)")
    parser.add_argument("--num-test-samples", type=int, default=10000, metavar="N", # ** 这里的num-test-samples是测试集的数量，默认是10000
                    help="number of testing samples to use (default: 10000)")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available() 
    print("Using Cuda is:", use_cuda)
    torch.manual_seed(args.seed)    

    device = torch.device("cuda" if use_cuda else "cpu")
    
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),          
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    dataset2 = datasets.MNIST("./data", train=False, download=True,
                            transform=transform)
    
    if args.num_test_samples < len(dataset2):
        dataset2 = Subset(dataset2, range(args.num_test_samples))

    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model_name = args.model.lower()
    if model_name == "lenet":
        model = LeNet().to(device)
    elif model_name == "defaultnet":
        model = DefaultNet().to(device)
    else:
        print("Wrong model name. Try again!")
        exit()
    
    model_path = Path("./model/weights/{}.pt".format(model_name))
    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
    else:
        print("Wrong model name. Try again!")
        exit()
    print("\nTest model:\t{}".format(args.model))
    test(model, device, test_loader)


if __name__ == "__main__":
    main()