clear;clc;

disp('��ȡѵ��ͼ��')
input0=load('train_image_mnist.mat');
train_image=input0.train_image;

disp('��ȡѵ����ǩ')
input0=load('train_label_mnist.mat');
train_label=input0.train_label;

disp('��ȡ����ͼ��')
input0=load('test_image_mnist.mat');
test_image=input0.test_image;

disp('��ȡ���Ա�ǩ')
input0=load('test_label_mnist.mat');
test_label=input0.test_label;

%ͨ����Ϊ1������˽�Ϊ3*3��ͼ��߳�27��10�ֱ�ǩ,3��
%net=setparameter(1,3,27,10,4);
%savenet('net_mnist.mat',net);
disp('��ȡ����')
net=load('net_mnist.mat');

%net_pre=load('net_mnist_pre.mat');
%net.layer=net_pre.layer;
%����ѵ���������磬����ѵ��
net=trainnet(net,train_image,train_label);



%����ѵ���õ������Լ����Լ������в���
%testnet(net,test_image,test_label);
