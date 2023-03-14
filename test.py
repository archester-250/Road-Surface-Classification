import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from model import resnet50
import torch.utils.data as Data
from torch.utils.data import DataLoader

if __name__ == '__main__':
    images = []
    truth = []
    path = 'data/resized'
    print('importing data...')
    y = 0
    for root, dirs, files in os.walk(path):
        for d in dirs:
            print(d)
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
    dataset = Data.TensorDataset(torch.Tensor(images), torch.Tensor(truth))
    loader = DataLoader(dataset=dataset, batch_size=50, shuffle=True)
    model = resnet50().to(device='cpu')
    model.load_state_dict(torch.load('model_lr_0.2.pth'), strict=False)
    avg_acc = []
    for i_batch, (image, label) in tqdm(enumerate(loader)):
        y = model(image)
        pred_prob, pred_label = torch.max(torch.Tensor(y), dim=1)
        acc = (pred_label == label).sum().item() * 1.0 / label.shape[0]
        print('Accuracy: %f' % acc)
        avg_acc.append(acc)
    print('Avg acc: %f' % np.mean(avg_acc))
