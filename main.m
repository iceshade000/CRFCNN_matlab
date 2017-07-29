clear;clc;

disp('¶ÁÈ¡ÑµÁ·Í¼Ïñ')
input=load('train_image_mnist.mat');
train_image=input.train_image;

disp('¶ÁÈ¡ÑµÁ·±êÇ©')
input=load('train_label_mnist.mat');
train_label=input.train_label;

disp('¶ÁÈ¡²âÊÔÍ¼Ïñ')
input=load('test_image_mnist.mat');
test_image=input.test_image;

disp('¶ÁÈ¡²âÊÔ±êÇ©')
input=load('test_label_mnist.mat');
test_label=input.test_label;

disp('¶ÁÈ¡ÍøÂç')
net=load('net_mnist.mat');

%ÊäÈëÑµÁ·¼¯ºÍÍøÂç£¬½øĞĞÑµÁ·
net=trainnet(net,train_image,train_label);

%ÊäÈëÑµÁ·ºÃµÄÍøÂçÒÔ¼°²âÊÔ¼¯£¬½øĞĞ²âÊÔ
%file='./dataset/net10000.mat';
%test(file,test_image,test_label);
