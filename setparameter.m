%%
%输入：tunnel_num通道数,kernel_len卷积核边长,image_len图像边长，kernel_num标签种数

%输出：net是一个结构，有layer,result,back三个部分
%这里的layer是一个cell，其中有4个cell,代表4个层，各个层的大小分别为
%10*1*3*3，10*10*3*3+1*10*10,10*10*3*3+2*10*10,10*10*3*3+3*10*10
%此处默认为随机数
%这里的result是一个cell，也有4个cell，代表4个层的输出，各个输出大小分别为
%10*27*27，10*9*9，10*3*3，10
%此处默认为0
%back用来存储反向传播的结果，与result一样大
%%
function [net]= setparameter(tunnel_num,kernel_len, image_len, kernel_num)
disp('设置网络中，请稍候')
%利用输入图像大小，输出标签数目，生成网络
%层数为图像边长对卷积核边长取log再+1
layer_num= round(log(image_len)/log(kernel_len))+1;
layer=cell(1,layer_num);
result=cell(1,layer_num);
%每层卷积核数目为标签数目
%第一层特殊处理一下
layer{1}=cell(1,tunnel_num);
for i=1:tunnel_num
    layer{1}{i}=rand(kernel_num,kernel_len,kernel_len);
end

%后面的层统一处理
for i=2:layer_num
    layer{i}=cell(1,2);
    layer{i}{1}=rand(kernel_num,kernel_num,kernel_len,kernel_len);
    layer{i}{2}=rand(i-1,kernel_num,kernel_num);
end

%设置好各层输出
temp=image_len;
for i=1:layer_num
    result{i}=zeros(kernel_num,temp,temp);
    temp=floor(temp/kernel_len);
end
back=result;

net.layer=layer;
net.result=result;
net.back=back;
end