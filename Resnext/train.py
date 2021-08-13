import os.path

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from torchsummary import summary as summary_
from models import ResNext50_32x4d
from tensorboardX import SummaryWriter

def accuracy(output, target, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

model = ResNext50_32x4d()
model = model.to(device)

summary_(model,(3,32,32),batch_size = 1)

model_name = "ResNext50_32x4d"
n_epoch = 300
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9,
                weight_decay=0.1)

decay_epoch = [60, 120, 160]
step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=decay_epoch, gamma=0.2)

writer = SummaryWriter("./logs")

def train():
    min_loss = 10**9

    for epoch in range(1, n_epoch + 1):
        model.train()
        train_loss = 0

        train_acc_top1 = 0
        train_acc_top5 = 0
        for batch_idx, samples in enumerate(trainloader):
            x_train, y_train = samples
            x_train, y_train = x_train.to(device), y_train.to(device)
            #print("eok : ",x_train.shape)

            prediction = model(x_train)
            loss = criterion(prediction, y_train)
            acc = accuracy(prediction, y_train)
            train_acc_top1 += acc[0].item()
            train_acc_top5 += acc[1].item()

            train_loss += loss.item()
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        train_acc_top1 = train_acc_top1/len(trainloader)
        train_acc_top5 = train_acc_top5 / len(trainloader)
        train_loss = train_loss/len(trainloader)

        model.eval()
        val_loss = 0
        val_acc_top1 = 0
        val_acc_top5 = 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

                acc = accuracy(output, target)
                val_acc_top1 += acc[0].item()
                val_acc_top5 += acc[1].item()

        val_loss = val_loss / len(testloader)
        val_acc_top1 = val_acc_top1 / len(testloader)
        val_acc_top5 = val_acc_top5 / len(testloader)

        print(f'epoch : {epoch} train loss : {train_loss : 0.4f} validation loss : {val_loss : 0.4f}')
        print(f'train top1 acc : {train_acc_top1 : 0.4f} train top5 acc : {train_acc_top5 : 0.4f}')
        print(f'val top1 acc : {val_acc_top1 : 0.4f} val top5 acc : {val_acc_top5 : 0.4f}')

        ### save model ###
        if val_loss < min_loss:
            min_loss = val_loss
            state = {
                'model': model.state_dict(),
                'tain_loss': train_loss,
                'val_loss': val_loss,
                'epoch' : epoch
            }
            if not os.path.isdir("model_save"):
                os.mkdir("model_save")
            torch.save(state, "./model_save/" + model_name + '.pth')
            print( "model is saved!!!", epoch)

        ### write tensorboard ###
        writer.add_scalars('loss/loss',{"train_loss": train_loss, "val_loss":val_loss},epoch)
        writer.add_scalars('accuracy/top 1',{"train top1 acc": train_acc_top1, "val top1 acc":val_acc_top1},epoch)
        writer.add_scalars('accuracy/top 5', {"train top5 acc": train_acc_top5, "val top5 acc": val_acc_top5}, epoch)

        writer.flush()

train()
writer.close()





