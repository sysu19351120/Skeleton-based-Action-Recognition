dataset.py是用于读取npy数据、数据预处理作传入网络准备的
dcgan.py用于读取npy数据并用CGAN生成数据和标签
plot.py用于绘制准确率曲线
show.py用于可视化生成序列数据
main.py用于训练stgcn
**********************************************************************
net文件夹中stgcn（包括utils文件夹内的工具文件）为本组主要实验网络，LSTM文件夹与MSCNN文件夹中为对应的另外两个实验网络
其中stgcn和MSCNN内有data文件夹可以直接跑看输入输出大小，LSTM中无data文件夹，要验证可将data文件夹移入LSTM文件夹中
data文件夹为数据集
gen_data文件夹为生成数据集，方便老师可视化起见训练后将其移出data单独生成成一个文件夹
checkpoint内为训练好的最终版本即效果最好的一个stgcn网络，运行main.py即可
**********************************************************************
test.txt为每个epoch用时以及测试集平均识别率和loss的记录
**********************************************************************
由于dcgan的模型（D_Net和G_Net）过大（超过800M）无法放入github，如需验证请直接运行dcgan.py训练

