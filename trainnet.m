%%
%训练网络
%输入：网络net,训练图像train_image,训练标签train_label
%%
function [net]=trainnet(net,train_image,train_label)
%获取图像数目，卷积核数目
[image_num,label_num]=size(train_label);
%迭代训练1000000次
iternum=1;

%训练函数
for i=1:iternum
    %从图像中随机选取一个,以后可以改进成随机选取一块
    temp=randi(image_num+1);
    input_image=squeeze(train_image{1}(temp,:,:));
    
    %前向计算，每过1000次输出总误差（均方误差）
    net.result=forward(net,input_image);
    
    %反向传播，更新参数的值
    input_label=train_label(temp,:);
    mu=0.0001;
    net=backward(net,input_label,input_image,mu);
    
    if mod(i,1000)==1
        t=net.back{size(net.back,2)};
        disp(dot(t,t'));
    end
    %每过10000次，保存网络
end
end