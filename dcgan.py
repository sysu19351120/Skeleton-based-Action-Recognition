import torch
from torch import nn
from torch.autograd import Variable
from dataset import GetData
import numpy as np
from tool.visualise import visualise
from tool.graph import Graph
bce_loss=nn.BCEWithLogitsLoss()
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3, 32, (3, 1), 1),
            nn.Conv2d(32, 32, (3, 1), 1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, (3, 1), 1),
            nn.Conv2d(64, 64, (3, 1), 1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
        )
        self.fc=nn.Sequential(
            nn.LeakyReLU(0.01),
            nn.Linear(64 * 47 * 8,5)
        )
    def forward(self,x):
        x=x.view(x.shape[0],3,200,17*2)
        x=self.conv(x) # x:b*64*47*8
        x=x.view(x.shape[0],-1)
        x=self.fc(x)
        return x
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
class Generator(nn.Module):
    def __init__(self,input_dim):
        super(Generator,self).__init__()
        self.nd=input_dim
        self.fc=nn.Sequential(
            nn.Linear(input_dim,1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,100*17*128),
            nn.ReLU(True),
            nn.BatchNorm1d(100*17*128)
        )
        # 更改
        '''self.fc = nn.Sequential(
            nn.Linear(input_dim, 100 * 17 * 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(100 * 17 * 128)
        )'''
        self.conv=nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,3,3,1,padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(3),
        )
        # 更改
        '''self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 2, 2, padding=1),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(3, 3, 2, 1),
            nn.ConvTranspose2d(3, 3, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )'''
    def forward(self,x):
        x=self.fc(x)
        x=x.view(x.shape[0],128,100,17)
        x=self.conv(x)
        x=x.view(x.shape[0],3,200,17,2)
        return x
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
def discriminator_loss(logits_real,logits_fake):
    size=logits_real.shape[0]
    true_labels=Variable(torch.ones(size,1)).float()
    false_labels=Variable(torch.zeros(size,1)).float()
    loss=bce_loss(logits_real,true_labels)+bce_loss(logits_fake,false_labels)
    return loss
def generator_loss(logits_fake):
    size=logits_fake.shape[0]
    true_labels=Variable(torch.ones(size,1)).float()
    loss=bce_loss(logits_fake,true_labels)
    return loss
def gen_data():
    D_net=Discriminator()
    G_net=Generator(input_dim=101)# 96+5
    try:
        D_checkpoint = torch.load('./checkpoint/D_net.t7')
        D_net.load_state_dict(D_checkpoint['state'])
        G_checkpoint = torch.load('./checkpoint/G_net.t7')
        G_net.load_state_dict(G_checkpoint['state'])
        print('===> Load last checkpoint data')
    except FileNotFoundError:
        print('Can\'t found model')
    num=100
    print('===> Start Generating...')
    for classi in range(5):
        sample_noise = (torch.rand(num, 96) - 0.5) / 0.5  # -1 ~ 1 的均匀分布
        labels=np.zeros((num,5))
        labels[:,classi]=1
        sample_noise = np.concatenate((sample_noise.numpy(), labels), 1)
        sample_noise = Variable(torch.from_numpy(sample_noise).float())
        fake_images=G_net(sample_noise)
        output = D_net(fake_images)
        for i in range(num):
            print(output[i])
            sample=torch.unsqueeze(fake_images[i,:,:,:,:],dim=0).detach()*200
            visualise(sample, graph=Graph(), is_3d=True)
def train():
    train_set=GetData('./data/train/',num=410,mode='train')
    train_loader=torch.utils.data.DataLoader(
        train_set,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    D_net=Discriminator().cuda()
    D_net.initialize_weights()
    G_net=Generator(input_dim=101).cuda() # 96+5
    G_net.initialize_weights()
    G_optimizer = torch.optim.Adam(G_net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    D_optimizer = torch.optim.Adam(D_net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    #G_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(G_optimizer, 'min',factor=0.1)
    #D_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(D_optimizer, 'min',factor=0.1)
    start_epoch = 0
    print('===> Try resume from checkpoint')
    try:
        D_checkpoint = torch.load('./checkpoint/D_net.t7')
        D_net.load_state_dict(D_checkpoint['state'])
        G_checkpoint = torch.load('./checkpoint/G_net.t7')
        G_net.load_state_dict(G_checkpoint['state'])
        start_epoch = D_checkpoint['epoch']
        print('===> Load last checkpoint data')
    except FileNotFoundError:
        print('Can\'t found model')
    print('===> Start Traning...')
    for epoch in range(start_epoch,30000):
        D_loss=0.0
        G_loss=0.0
        D_net.train()
        G_net.train()
        for i,(x,label) in enumerate(train_loader):
            #print(x)
            bs = x.shape[0]
            labels_onehot = np.zeros((bs, 5))
            labels_onehot[np.arange(bs), label.int().numpy()] = 1
            real_label = Variable(torch.from_numpy(labels_onehot).float()).cuda()  # 真实label为1
            fake_label = Variable(torch.zeros(bs, 5)).cuda()  # 假的label为0
            # 判别网络
            real_data = Variable(x).cuda()  # 真实数据
            logits_real = D_net(real_data)  # 判别网络得分
            d_loss_real = bce_loss(logits_real, real_label)
            real_scores = logits_real
            sample_noise = (torch.randn(bs, 101) - 0.5) / 0.5  # -1 ~ 1 的均匀分布
            sample_noise=sample_noise.cuda()
            g_fake_seed = Variable(sample_noise)
            fake_images = G_net(g_fake_seed)  # 生成的假的数据
            logits_fake = D_net(fake_images)  # 判别网络得分
            d_loss_fake=bce_loss(logits_fake,fake_label)
            fake_scores = logits_fake
            d_total_error = d_loss_real + d_loss_fake
            D_optimizer.zero_grad()
            d_total_error.backward()
            D_optimizer.step()  # 优化判别网络

            # 生成网络
            sample_noise = (torch.rand(bs, 96) - 0.5) / 0.5  # -1 ~ 1 的均匀分布
            g_fake_seed = Variable(sample_noise)
            g_fake_seed=np.concatenate((g_fake_seed.numpy(),labels_onehot),axis=1)
            g_fake_seed = Variable(torch.from_numpy(g_fake_seed).float()).cuda()
            fake_images = G_net(g_fake_seed)  # 生成的假的数据

            gen_logits_fake = D_net(fake_images)
            g_error = bce_loss(gen_logits_fake,real_label)  # 生成网络的 loss
            G_optimizer.zero_grad()
            g_error.backward()
            G_optimizer.step()  # 优化生成网络
            D_loss+=d_total_error.item()
            G_loss += g_error.item()
            if i%5==4:
                print('epoch:',epoch+1," D_loss:",D_loss/5," G_loss:",G_loss/5)
                #G_schedule.step(G_loss/5)
                #D_schedule.step(D_loss/5)
                D_loss = 0.0
                G_loss = 0.0
        print('===> Saving models...')
        D_state = {
            'state': D_net.state_dict(),
            'epoch': epoch
        }
        torch.save(D_state, './checkpoint/D_net.t7')
        G_state = {
            'state': G_net.state_dict(),
            'epoch': epoch
        }
        torch.save(G_state, './checkpoint/G_net.t7')
        print('===> Saving Completed!!')
if __name__=="__main__":
    '''t=torch.rand((8,3,200,17,2))
    noise=torch.rand((32,96))
    g_net=Generator(input_dim=96)
    d_net=Discriminator()
    g_net.initialize_weights()
    output=g_net(noise)
    #output = d_net(t)
    print(output.shape)'''
    gen_data()
    #train()
