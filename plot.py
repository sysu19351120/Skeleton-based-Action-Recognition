import matplotlib.pyplot as plt
f=open('test.txt')
lines=f.readlines()
y=[]
x=[]
for i,line in enumerate(lines):
    start_pos=line.find('acc:')+4
    acc=float(line[start_pos:start_pos+7])
    y.append(acc)
    x.append(i)
plt.plot(x, y)
plt.show()
'''squares=[1, 4, 9, 16, 25]
x=[1, 2, 3, 4, 5]
plt.plot(x, squares)
plt.show()'''
'''label=torch.rand((1,3,200,17,2))*random.randint(-300,300)
print(torch.max(label))
print(torch.min(label))
print(abs(torch.min(label).item()))'''
'''output=nn.Conv2d(3,3,kernel_size=(3,1),padding=(4,0))(label)
output=nn.Conv2d(3,3,kernel_size=(3,1))(output)
output=nn.Conv2d(3,3,kernel_size=(3,1))(output)
output=nn.Conv2d(3,3,kernel_size=(3,1))(output)
print(output.shape)'''
