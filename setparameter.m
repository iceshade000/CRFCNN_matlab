%%
%���룺tunnel_numͨ����,kernel_len����˱߳�,image_lenͼ��߳���kernel_num��ǩ����

%�����net��һ���ṹ����layer,result,back,bias�ĸ�����
%�����layer��һ��cell��������4��cell,����4���㣬������Ĵ�С�ֱ�Ϊ
%10*1*3*3�����涼��10*10*3*3,����ȫ���Ӳ���10*10
%bias��һ��cell����4��cell������4���㣬������ֱ�Ϊ
%10*1�����涼��10*10
%�˴�Ĭ��Ϊ�����
%�����result��һ��cell��Ҳ��4��cell������4�������������������С�ֱ�Ϊ
%10*27*27��10*9*9��10*3*3��10
%�˴�Ĭ��Ϊ0
%back�����洢���򴫲��Ľ������resultһ����
%%
function [net]= setparameter(tunnel_num,kernel_len, image_len, kernel_num,layer_num)
disp('���������У����Ժ�')
%��������ͼ���С�������ǩ��Ŀ����������
%����Ϊͼ��߳��Ծ���˱߳�ȡlog��+1
layer=cell(1,layer_num);
result=cell(1,layer_num);
bias=cell(1,layer_num);
%ÿ��������ĿΪ��ǩ��Ŀ
%��һ�����⴦��һ��
layer{1}=rand(kernel_num,tunnel_num,kernel_len,kernel_len);
bias{1}= rand(kernel_num,tunnel_num);
%����Ĳ�ͳһ����
for i=2:layer_num-1
    layer{i}=rand(i*kernel_num,(i-1)*kernel_num,kernel_len,kernel_len);
    bias{i}=rand(i*kernel_num,(i-1)*kernel_num);
end
%���һ��maxpooling+crfprob�ټӸ�ȫ���Ӳ㣬��˲���Ϊ10*10
layer{layer_num}=rand(kernel_num,(layer_num-1)*kernel_num);

%���úø������
temp=image_len;
for i=1:layer_num-1
    result{i}=zeros(i*kernel_num,temp,temp);
    temp=floor(temp/kernel_len);
end
%���һ�����ֱ�ӱ�Ϊ10*1*1
result{layer_num}=zeros(kernel_num,1,1);
back=result;

net.layer=layer;
net.result=result;
net.back=back;
net.bias=bias;
end