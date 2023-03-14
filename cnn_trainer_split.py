import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from model import resnet50
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    if args.pretrain is True:
        path = 'data/resized'
        if not os.path.exists(path):
            os.mkdir(path)
        path1 = 'data/basic_data'
        print('preprocessing data...')
        for root, dirs, files in os.walk(path1):
            for d in dirs:
                if not os.path.exists(os.path.join(path, d)):
                    os.mkdir(os.path.join(path, d))
                files1 = os.listdir(os.path.join(path1, d))
                for file1 in tqdm(files1):
                    im = Image.open(os.path.join(path1, d, file1))
                    im = im.resize((224, 224))
                    im.save(os.path.join(path, d, file1))
        path2 = 'data/test_resized'
        if not os.path.exists(path2):
            os.mkdir(path2)
        test_path = 'data/pred_data'
        files = os.listdir(test_path)
        for file in tqdm(files):
            im = Image.open(os.path.join(test_path, file))
            im = im.resize((224, 224))
            im.save(os.path.join(path2, file))
        print('preprocess success!')
    if args.train is True:
        images = []
        truth = []
        y = 0
        path = 'data/resized'
        print('importing data...')
        for root, dirs, files in os.walk(path):
            for d in dirs:
                files1 = os.listdir(os.path.join(path, d))
                for file1 in tqdm(files1):
                    img = plt.imread(os.path.join(path, d, file1))
                    img = img.transpose((2, 0, 1))
                    images.append(img)
                    truth.append(y)
                y += 1
        images = np.array(images)
        truth = np.array(truth)
        print(images.shape)
        print('import data success!')
        images_train, images_val, truth_train, truth_val = train_test_split(images, truth, test_size=0.1)
        dataset_train = Data.TensorDataset(torch.Tensor(images_train), torch.Tensor(truth_train))
        train(dataset_train)
        dataset_val = Data.TensorDataset(torch.Tensor(images_val), torch.Tensor(truth_val))
        test(dataset_val)


def train(dataset):
    device = torch.device('cpu' if args.no_cuda else 'cuda')

    model = resnet50().to(device=device)
    if os.path.exists('model.pth'):
        model.load_state_dict(torch.load('model.pth'), strict=False)
    if not args.no_cuda:
        model = nn.DataParallel(model)

    opt = optim.SGD(model.parameters(), lr=args.lr)
    print('learning rate:%f' % args.lr)

    scheduler = CosineAnnealingLR(opt, args.epochs)

    loss_function = nn.CrossEntropyLoss()
    best_acc = 0.

    for epoch in range(args.epochs):
        model.train()
        scheduler.step()
        total_loss = 0.
        train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
        for i_batch, (image, label) in tqdm(enumerate(train_loader)):
            y = model(image)
            label = torch.LongTensor(label.view(-1).numpy())
            loss = loss_function(y, label)
            total_loss += loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            pred_prob, pred_label = torch.max(y, dim=1)
            acc = (pred_label == label).sum().item() * 1.0 / label.shape[0]
            print('Epoch : %d, Batch : %d, Loss : %f, Batch Accuracy %f' % (epoch, i_batch, loss, acc))
            if best_acc <= acc:
                best_acc = acc
                print('saving weights...')
                torch.save(model.state_dict(), 'model.pth')


def test(dataset):
    total_loss = 0.
    device = torch.device('cpu' if args.no_cuda else 'cuda')
    model = resnet50().to(device=device)

    if os.path.exists('model.pth'):
        model.load_state_dict(torch.load('model.pth'), strict=False)
    model.eval()
    val_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    avg_acc = []
    for i_batch, (image, label) in tqdm(enumerate(val_loader)):
        y = model(image)
        label = torch.LongTensor(label.view(-1).numpy())
        pred_prob, pred_label = torch.max(y, dim=1)
        acc = (pred_label == label).sum().item() * 1.0 / label.shape[0]
        print('Batch : %d, Batch Accuracy %f' % (i_batch, acc))
        avg_acc.append(acc)
    print('Avg acc: %f' % np.mean(avg_acc))


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=True, help='train or test')
    parser.add_argument('--pretrain', type=bool, default=False, help='preprocess pictures or not')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--no_cuda', type=bool, default=True, help='GPU or not')
    parser.add_argument('--epochs', type=int, default=10, help='epochs of train')
    # parser.add_argument()
    args = parser.parse_args()
    main()
