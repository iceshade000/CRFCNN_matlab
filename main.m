clear;clc;

disp('��ȡѵ��ͼ��')
input=load('train_image_mnist.mat');
train_image=input.train_image;

disp('��ȡѵ����ǩ')
input=load('train_label_mnist.mat');
train_label=input.train_label;

disp('��ȡ����ͼ��')
input=load('test_image_mnist.mat');
test_image=input.test_image;

disp('��ȡ���Ա�ǩ')
input=load('test_label_mnist.mat');
test_label=input.test_label;

disp('��ȡ����')
net=load('net_mnist.mat');

%����ѵ���������磬����ѵ��
net=trainnet(net,train_image,train_label);

%����ѵ���õ������Լ����Լ������в���
%file='./dataset/net10000.mat';
%test(file,test_image,test_label);
