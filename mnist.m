clear;clc;

%��ȡѵ��ͼ��
str0='./dataset/train-images.idx3-ubyte';
train_image_raw=loadMNISTImages(str0);
%size(train_image_raw)%784 60000
%��ȡѵ����ǩ
str1='./dataset/train-labels.idx1-ubyte';
train_label_raw=loadMNISTLabels(str1);
%size(train_label)%60000 1
%��ȡ����ͼ��
str2='./dataset/t10k-images.idx3-ubyte';
test_image_raw=loadMNISTImages(str2);
%size(test_image_raw)%784 10000
%��ȡ���Ա�ǩ
str3='./dataset/t10k-labels.idx1-ubyte';
test_label_raw=loadMNISTLabels(str3);
%size(test_label)%10000 1


%���ú�����ͼ��Ĵ�С��������27*27�������ǩ����Ŀ��������10����������
%ͼ��ת��Ϊcell�ͣ�Ϊ�Ժ��ͨ��ͼ��춨����
 train_image={rand(60000,27,27)};
 for i=1:60000
     for j=1:27
         for k=1:27
             train_image{1}(i,j,k)=train_image_raw((j-1)*28+k,i);
         end
     end
 end
 
 test_image={rand(10000,27,27)};
 for i=1:10000
     for j=1:27
         for k=1:27
             test_image{1}(i,j,k)=test_image_raw((j-1)*28+k,i);
         end
     end
 end
 disp('����ͼ����')
 save('train_image_mnist.mat','train_image');
 save('test_image_mnist.mat','test_image');
 
 %�����ǩ��ʹ���Ϊone hot����
 train_label=zeros(60000,10);
 test_label=zeros(10000,10);
 for i=1:60000
     train_label(i,train_label_raw(i)+1)=1;
 end
 for i=1:10000
     test_label(i,test_label_raw(i)+1)=1;
 end
 disp('�����ǩ��')
 save('train_label_mnist.mat','train_label');
 save('test_label_mnist.mat','test_label');
%ͨ����Ϊ1������˽�Ϊ3*3��ͼ��߳�27��10�ֱ�ǩ,3��
net=setparameter(1,3,27,10,4);
savenet('net_mnist.mat',net);



