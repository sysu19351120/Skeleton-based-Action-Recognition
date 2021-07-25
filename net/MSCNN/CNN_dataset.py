import torch
import numpy as np
import glob
import random
from tool.visualise import visualise
from tool.graph import Graph
from torchvision import transforms
class GetData(torch.utils.data.Dataset):
    def __init__(self,filedir,num,mode):
        classis=['000','001','002','003','004']
        self.inp=torch.zeros((num,3,200,17,2))
        self.inp_time = torch.zeros((num, 3, 200, 17, 2))
        self.inp_pos = torch.zeros((num, 3, 200, 17, 2))
        self.lab=torch.zeros(num)
        count=0
        for i,classi in enumerate(classis):
            paths=glob.glob(filedir+classi+'/*.npy')
            for sample_path in paths:
                #sample_path = './data/train/000/P000S00G10B10H50UC022000LC021000A000R0_08251609.npy'
                sample = np.load(sample_path)
                sample[:,1,:,:,:] = 0 #torch.Size([1, 3, 128, 17, 2])
                if sample.shape[2]<50:
                    sample=np.concatenate((sample,sample),axis=2)
                if sample.shape[2]<100:
                    sample=np.concatenate((sample,sample),axis=2)
                if sample.shape[2]< 200:
                    t=sample[:,:,:200-sample.shape[2],:,:]
                    sample = np.concatenate((sample,t), axis=2)
                else:
                    sample=sample[:,:,:200,:,:]
                # 动作加噪
                p = random.random()
                if p > 0.7:
                    error = 0.5
                    noise = error * np.random.normal(size=sample.shape)
                    sample = sample + noise
                # 倒序动作
                p = random.random()
                if p > 0.7:
                    sample = np.flip(sample, 2).copy()
                p = random.uniform(1, 2)
                sample/=p
                sample=torch.from_numpy(sample)
                '''*****************************************************'''
                '''max=torch.max(sample).item()
                min=abs(torch.min(sample).item())
                if max>=min:                                 # cGAN专用，main中删去！！！
                    sample/=max
                else:
                    sample/=min'''
                '''*****************************************************'''
                #visualise(sample, graph=Graph(), is_3d=True)
                #print(sample)
                self.inp[count]=sample
                self.lab[count]=i
                count+=1
        #print(self.inp)
        for iter in range( 200):
            self.inp_time[:, :, iter ,:] = self.inp[:, :, iter, :] - self.inp[:, :, iter - 1, :]
            if i < 4:
                self.inp_pos[i - 1, :, :, :] = self.inp[iter, :, :, :] - self.inp[iter - 1, :, :, :]
        '''self.inp_time[:, :, 199, :] = self.inp[:, :, 0, :] - self.inp[:, :, 199, :]
        self.inp_pos[num-1, :, :, :] = self.inp[0, :, :, :] - self.inp[num-1, :, :, :]'''
        #print(input.shape)
        #print(self.lab)
    def __getitem__(self, item):
        return self.inp[item],self.inp_time[item],self.inp_pos[item],self.lab[item]
    def __len__(self):
        return self.inp.size(0)
if __name__=="__main__":
    train_set=GetData('./data/train/',num=410,mode='train')
    train_loader=torch.utils.data.DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )
    for epoch in range(1000):
        for i,data in enumerate(train_loader):
            x=0