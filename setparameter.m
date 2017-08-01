%%
%输入：tunnel_num通道数,kernel_len卷积核边长,image_len图像边长，kernel_num标签种数

%输出：net是一个结构，有layer,result,back,bias四个部分
%这里的layer是一个cell，其中有4个cell,代表4个层，各个层的大小分别为
%10*1*3*3，后面都是10*10*3*3,最后的全连接层是10*10
%bias是一个cell，有4个cell，代表4个层，各个层分别为
%10*1，后面都是10*10
%此处默认为随机数
%这里的result是一个cell，也有4个cell，代表4个层的输出，各个输出大小分别为
%10*27*27，10*9*9，10*3*3，10
%此处默认为0
%back用来存储反向传播的结果，与result一样大
%%
function [net]= setparameter(tunnel_num,kernel_len, image_len, kernel_num,layer_num)
disp('设置网络中，请稍候')
%利用输入图像大小，输出标签数目，生成网络
%层数为图像边长对卷积核边长取log再+1
layer=cell(1,layer_num);
result=cell(1,layer_num);
bias=cell(1,layer_num);
%每层卷积核数目为标签数目
%第一层特殊处理一下
layer{1}=rand(kernel_num,tunnel_num,kernel_len,kernel_len);
bias{1}= rand(kernel_num,tunnel_num);
%后面的层统一处理
for i=2:layer_num-1
    layer{i}=rand(i*kernel_num,(i-1)*kernel_num,kernel_len,kernel_len);
    bias{i}=rand(i*kernel_num,(i-1)*kernel_num);
end
%最后一层maxpooling+crfprob再加个全连接层，因此参数为10*10
layer{layer_num}=rand(kernel_num,(layer_num-1)*kernel_num);

%设置好各层输出
temp=image_len;
for i=1:layer_num-1
    result{i}=zeros(i*kernel_num,temp,temp);
    temp=floor(temp/kernel_len);
end
%最后一层输出直接变为10*1*1
result{layer_num}=zeros(kernel_num,1,1);
back=result;

net.layer=layer;
net.result=result;
net.back=back;
net.bias=bias;
end