%%
%训练网络
%输入：网络net,训练图像train_image,训练标签train_label
%%
function [net]=trainnet(net,train_image,train_label)
%获取图像数目，卷积核数目
[image_num,label_num]=size(train_label);
%迭代训练1000000次
iternum=1000000;
mu=10;
all=0;
%训练函数
for i=1:iternum
    %从图像中随机选取一个,以后可以改进成随机选取一块
    temp=mod(randi(image_num)+30000,image_num)+1;
    input_image=train_image{1}(temp,:,:);
    
    %前向计算，每过1000次输出总误差（均方误差）
    net=forward(net,input_image);
    
    %反向传播，更新参数的值
    input_label=train_label(temp,:);
    net=backward(net,input_label,input_image,mu);
    %如果出错了就再跑一次
    t=net.back{size(net.back,2)};
    if dot(t,t')>0.5 %经过计算可知，小于0.5肯定没错
       % net=forward(net,input_image);
        all=all+1;
        %net=backward(net,input_label,input_image,mu);
    end
    
    if mod(i,100)==1;
        disp(['错误数量：',num2str(all)]);
        all=0;
    end
 
    if mod(i,20)==1         
        
        disp(['第',num2str(i),'轮迭代: ']);
        net=forward(net,input_image);
        [ma,I]=max(squeeze(net.result{size(net.back,2)}));
        disp(['输入数字是',num2str(find(input_label)-1),' ,输出数字是',num2str(I-1)]);
        t2=input_label'-net.result{size(net.result,2)};
        disp(['误差为：',num2str(dot(t,t')),'->',num2str(dot(t2,t2'))]);
        disp(['部分参数：',num2str(net.layer{2}(5,5,2,3)),'偏置',num2str(net.bias{2}(5,5))]);
    end
    %每过1000次，保存网络
    if mod(i,1000)==0
        disp('保存网络');
        savenet('net_mnist.mat',net);
    end
end
end