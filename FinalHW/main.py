import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from net.st_gcn import Model
from dataset import GetData
from sklearn.metrics import accuracy_score

def accuracy(preds, target):
    preds = torch.max(preds, 1)[1].float()
    acc = accuracy_score(preds.cpu().numpy(), target.cpu().numpy())
    return acc
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
    model.eval()
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        outputs = model(inputs)
        #preds = torch.max(outputs, 1)[1].int()
        #print(preds.item(),'<---->',labels.int().item())
        loss = criterion(outputs, labels.long())
        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)
    print('Test--->loss:', running_loss / 117, ' accuracy:', running_acc / 117)
    f=open('test.txt','a')
    f.write(' loss:{} acc:{} \n'.format(running_acc/117,running_acc/117))
    f.close()
    return running_acc / 117

if __name__=="__main__":
    train_set=GetData('./data/train/',num=410,mode='train')
    train_loader=torch.utils.data.DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )
    model=Model(3,5,'sptial',True).cuda()
    model.initialize_weights()
    start_epoch = 0
    print('===> Try resume from checkpoint')
    try:
        checkpoint = torch.load('./checkpoint/model.t7')
        model.load_state_dict(checkpoint['state'])
        start_epoch = checkpoint['epoch']
        print('===> Load last checkpoint data')
    except FileNotFoundError:
        print('Can\'t found model.t7')
    print('===> Start Traning...')
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.01,weight_decay=0.001)
    schedule=optim.lr_scheduler.StepLR(optimizer,step_size=200,gamma=0.1)
    acc_list=[]
    for epoch in range(start_epoch,30000):
        since=time.time()
        running_loss=0.0
        running_acc=0.0
        model.train()
        for i,data in enumerate(train_loader):
            inputs,labels=data
            sample=inputs.numpy()
            # 动作加噪.

            '''p = random.random()
            if p > 0.5:
                error = 0.2
                noise = error * np.random.normal(size=sample.shape)
                sample = sample + noise
            # 倒序动作
            p = random.random()
            if p > 0.5:
                sample = np.flip(sample, 2).copy()'''
            inputs=torch.from_numpy(sample).float()
            inputs,labels=Variable(inputs).cuda(),Variable(labels).cuda()
            outputs=model(inputs)  #inputs:[8 ,3 ,200 ,17 ,2 ]
            #print(outputs,'<---->',labels.item())
            loss=criterion(outputs,labels.long())
            '''L1_reg = 0
            for param in model.parameters():
                L1_reg += torch.sum(torch.abs(param))
            loss += 0.001 * L1_reg '''
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            running_acc+=accuracy(outputs,labels)
            if i%10==9:
                print('[epoch:',epoch+1,' i:',i+1,"]--->loss:",running_loss/10,' accuracy:',running_acc/10)
                running_loss=0.0
                running_acc=0.0
        schedule.step()
        time_elapsed = time.time() - since
        f = open('test.txt', 'a')
        f.write('[epoch {} complete in {}s '.format(epoch,time_elapsed))
        f.close()
        print('===>Start Testing...')
        test_acc=test(model)
        acc_list.append(test_acc)
        if np.max(acc_list)==test_acc:
            print('===> Saving models...')
            state = {
                'state': model.state_dict(),
                'epoch': epoch
            }
            torch.save(state, './checkpoint/model.t7')
            print('===> Saving Completed!!')