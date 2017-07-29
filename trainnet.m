%%
%ѵ������
%���룺����net,ѵ��ͼ��train_image,ѵ����ǩtrain_label
%%
function [net]=trainnet(net,train_image,train_label)
%��ȡͼ����Ŀ���������Ŀ
[image_num,label_num]=size(train_label);
%����ѵ��1000000��
iternum=1;

%ѵ������
for i=1:iternum
    %��ͼ�������ѡȡһ��,�Ժ���ԸĽ������ѡȡһ��
    temp=randi(image_num+1);
    input_image=squeeze(train_image{1}(temp,:,:));
    
    %ǰ����㣬ÿ��1000���������������
    net.result=forward(net,input_image);
    
    %���򴫲������²�����ֵ
    input_label=train_label(temp,:);
    mu=0.0001;
    net=backward(net,input_label,input_image,mu);
    
    if mod(i,1000)==1
        t=net.back{size(net.back,2)};
        disp(dot(t,t'));
    end
    %ÿ��10000�Σ���������
end
end