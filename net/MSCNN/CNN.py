import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import time
from net.MSCNN.CNN_dataset import GetData
from sklearn.metrics import accuracy_score
def accuracy(preds, target):
    preds = torch.max(preds, 1)[1].float()
    acc = accuracy_score(preds.cpu().numpy(), target.cpu().numpy())
    return acc
class MSCNN(nn.Module):
    def __init__(self):
        super(MSCNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(3,64,1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            #nn.Dropout(0.5)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(34, 32, 3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            #nn.Dropout(0.5)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(3*64,256,3,stride=2,padding=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,512,3,stride=2,padding=2),
            nn.LeakyReLU(),
            #nn.Dropout(0.5)
        )
        self.meanpolling=nn.AvgPool2d(3,padding=1)
        self.maxpooling=nn.MaxPool2d(3,padding=1)
        self.fc=nn.Sequential(
            nn.Linear(512*2*5*2,512),
            #nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,5),
            nn.LeakyReLU()
        )
    def forward(self,x,x_time,x_pos):
        x,x_time,x_pos=x.view(x.shape[0],3,200,17*2),\
                       x_time.view(x_time.shape[0],3,200,17*2),x_pos.view(x_pos.shape[0],3,200,17*2)
        x=self.conv1(x)
        x=x.permute(0,3,2,1)
        x=self.conv2(x)
        x_time = self.conv1(x_time)
        x_time = x_time.permute(0, 3, 2, 1)
        x_time = self.conv2(x_time)
        x_pos = self.conv1(x_pos)
        x_pos = x_pos.permute(0, 3, 2, 1)
        x_pos = self.conv2(x_pos)
        output=torch.cat((x,x_time,x_pos),dim=1)
        output=self.conv3(output)
        output1=self.meanpolling(output)
        output2=self.maxpooling(output)
        output=torch.cat((output1,output2),dim=1)
        output=output.view(output.size(0),-1)
        output=self.fc(output)
        return output
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
def test(model):
    test_set = GetData('./data/test/',num=117,mode='test')
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    running_loss=0.0
    running_acc=0.0
    acc=0
    model.eval()
    for i, data in enumerate(test_loader):
        inputs,inputs_time,inputs_pos, labels = data
        inputs,inputs_time,inputs_pos, labels = Variable(inputs).cuda(), Variable(inputs_time).cuda(),Variable(inputs_pos).cuda(),Variable(labels).cuda()
        outputs = model(inputs,inputs_time,inputs_pos)
        preds = torch.max(outputs, 1)[1].int()
        #print(preds.item(),'<---->',labels.int().item())
        loss = criterion(outputs, labels.long())
        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)
        if preds.item()==labels.int().item():
            acc+=1
    print('Test--->loss:', running_loss / 117, ' accuracy:', running_acc / 117)
    f=open('test.txt','a')
    f.write(' loss:{} acc:{} \n'.format(running_acc/117,running_acc/117))
    f.close()
    print("real acc:",acc/117)
    return running_acc / 117
if __name__ == "__main__":
    train_set = GetData('./data/train/', num=410, mode='train')
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )
    model = MSCNN().cuda()
    model.initialize_weights()
    start_epoch = 0
    print('===> Try resume from checkpoint')
    try:
        checkpoint = torch.load('./checkpoint/MSCNN.t7')
        model.load_state_dict(checkpoint['state'])
        start_epoch = checkpoint['epoch']
        print('===> Load last checkpoint data')
    except FileNotFoundError:
        print('Can\'t found MSCNN.t7')
    print('===> Start Traning...')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,betas=[0.5,0.999])
    schedule = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    acc_list = []
    for epoch in range(start_epoch, 30000):
        since = time.time()
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            inputs, inputs_time, inputs_pos, labels = data
            inputs, inputs_time, inputs_pos, labels = Variable(inputs).cuda(), Variable(inputs_time).cuda(), Variable(inputs_pos).cuda(), Variable(labels).cuda()
            outputs = model(inputs, inputs_time, inputs_pos)  # inputs:[8 ,3 ,200 ,17 ,2 ]
            #print(outputs)
            # print(outputs,'<---->',labels.item())
            loss = criterion(outputs, labels.long())
            '''L1_reg = 0
            for param in model.parameters():
                L1_reg += torch.sum(torch.abs(param))
            loss += 0.001 * L1_reg '''
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += accuracy(outputs, labels)
            if i % 10 == 9:
                print('[epoch:', epoch + 1, ' i:', i + 1, "]--->loss:", running_loss / 10, ' accuracy:',
                      running_acc / 10)
                running_loss = 0.0
                running_acc = 0.0
        schedule.step()
        time_elapsed = time.time() - since
        f = open('test.txt', 'a')
        f.write('[epoch {} complete in {}s '.format(epoch,time_elapsed))
        f.close()
        print('===>Start Testing...')
        test_acc = test(model)
        acc_list.append(test_acc)
        if np.max(acc_list) == test_acc:
            print('===> Saving models...')
            state = {
                'state': model.state_dict(),
                'epoch': epoch
            }
            torch.save(state, './checkpoint/MSCNN.t7')
            print('===> Saving Completed!!')
'''a=torch.randn((4,3,200,34))
a_time=torch.zeros((4,3,200,34))
a_pos=torch.zeros((4,3,200,34))
for i in range(1,200):
    a_time[:,:,i-1,:]=a[:,:,i,:]-a[:,:,i-1,:]
    if i<4:
        a_pos[i-1,:,:,:]=a[i,:,:,:]-a[i-1,:,:,:]
a_time[:,:,199,:]=a[:,:,0,:]-a[:,:,199,:]
a_pos[3,:,:,:]=a[0,:,:,:]-a[3,:,:,:]
model=MSCNN()
output=model(a,a_time,a_pos)
#a,a_time,a_pos=torch.from_numpy(a),torch.from_numpy(a_time),torch.from_numpy(a_pos)
output=nn.Conv2d(3,64,(1,1))(a)
output=nn.Conv2d(64,64,(3,1))(output) #b*64*198*34
output=output.permute(0,3,2,1)
output=nn.Conv2d(34,32,3,stride=2,padding=2)(output)
output=nn.Conv2d(32,64,3,stride=2,padding=2)(output)
print(output.shape)'''
#output=nn.Conv3d(64,32,(3,3,1),stride=2,padding=2)(output)