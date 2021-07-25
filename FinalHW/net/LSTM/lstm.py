import os.path
import random
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch.nn.functional as F
import time
'''使用时请将data文件夹移入LSTM文件夹中！！！！不然无法读取数据！！！！'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_paths = glob.glob('./data/train/*')
test_paths = glob.glob('./data/test/*')
max_d = 301
HiddenSize = 60


def data_process(sample):
    p = random.random()
    if p > 0.7:
        error = 0.5
        noise = error * np.random.normal(size=sample.shape)
        sample = sample + noise
    # 倒序动作
    p = random.random()
    if p > 0.7:
        sample = np.flip(sample, 2).copy()
    return sample


def handle(sample):
    sample[:, 1, :, :, :] = 0
    sample = sample[:, [0, 2], :, :, :]
    return sample


def GetData(batch, path=train_paths, more=False):
    data = np.zeros((batch, 2, max_d, 17, 1))
    label = np.zeros((batch, 1))
    num = 0
    for i, file in enumerate(path):
        for sample_path in os.listdir(file):
            sample = np.load(os.path.join(file, sample_path))

            if more:
                more_sample = data_process(sample)
                more_sample = handle(more_sample)
                _, d2, d3, d4, d5 = sample.shape
                data[num, :d2, :d3, :d4, :] = more_sample[0, :, :, :, 0:1]
                label[num] = i
                num += 1

            sample = handle(sample)
            _, d2, d3, d4, d5 = sample.shape
            data[num, :d2, :d3, :d4, :] = sample[0, :, :, :, 0:1]
            label[num] = i
            num += 1
    return data.squeeze().reshape(batch, max_d, 34), label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm1 = nn.LSTM(input_size=34, hidden_size=HiddenSize, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=HiddenSize, hidden_size=HiddenSize, num_layers=1, batch_first=True)
        # 原来的输入格式是：(seq, batch, shape)，设置batch_first=True以后，输入格式就可以改为：(batch, seq, shape)
        self.linear1 = nn.Linear(HiddenSize * max_d, max_d)
        self.linear2 = nn.Linear(max_d, 5)

    def forward(self, x):
        x, (h, c) = self.lstm1(x)
        x, (h, c) = self.lstm2(x)
        x = x.reshape(-1, HiddenSize * max_d)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


loss_fun = nn.CrossEntropyLoss()
model_lstm = Net().to(device)
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.003, betas=(0.9, 0.999))

model_lstm.train()

train_set, label = GetData(820, more=True)

temp_train = list(zip(train_set, label))
random.shuffle(temp_train)
train_set[:], label[:] = zip(*temp_train)

train_set = torch.from_numpy(train_set).float().to(device)
label = torch.from_numpy(label).long().to(device)
label = label.squeeze(1)
visualization_loss = []

test_set, test_label = GetData(117, test_paths)
test_set = torch.from_numpy(test_set).float().to(device)
test_label = torch.from_numpy(test_label).float().to(device)
test_label = test_label.squeeze(1)

plot_acc = []
for epoch in range(1000):
    since=time.time()
    output = model_lstm(train_set)
    loss = loss_fun(output, label)
    optimizer.zero_grad()
    loss.backward()
    visualization_loss.append(loss.item())
    optimizer.step()
    test_output = model_lstm(test_set)
    test_result = torch.max(test_output, 1)[1]
    num = torch.sum(test_result == test_label)
    plot_acc.append(num / 117)
    time_elapsed = time.time() - since
    f = open('test.txt', 'a')
    f.write('[epoch {} complete in {}s '.format(epoch, time_elapsed))
    f.close()
    if epoch % 200 == 0 and epoch > 0:
        print("epoch:{}, loss:{}".format(epoch, loss.item()))

torch.save(model_lstm, "./model//model6.pkl")
visualization_loss = np.array(visualization_loss)
plt.figure()
plt.plot(range(1000), visualization_loss)
plt.figure()
plot_acc = torch.tensor(plot_acc)
plt.plot(range(1000), plot_acc)
print(plot_acc)
plt.show()
