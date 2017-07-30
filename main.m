clear;clc;

disp('读取训练图像')
input0=load('train_image_mnist.mat');
train_image=input0.train_image;

disp('读取训练标签')
input0=load('train_label_mnist.mat');
train_label=input0.train_label;

disp('读取测试图像')
input0=load('test_image_mnist.mat');
test_image=input0.test_image;

disp('读取测试标签')
input0=load('test_label_mnist.mat');
test_label=input0.test_label;

%通道数为1，卷积核皆为3*3，图像边长27，10种标签,3层
%net=setparameter(1,3,27,10,4);
%savenet('net_mnist.mat',net);
disp('读取网络')
net=load('net_mnist.mat');

%net_pre=load('net_mnist_pre.mat');
%net.layer=net_pre.layer;
%输入训练集和网络，进行训练
net=trainnet(net,train_image,train_label);



%输入训练好的网络以及测试集，进行测试
%testnet(net,test_image,test_label);
