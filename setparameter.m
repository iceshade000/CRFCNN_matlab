%%
%���룺tunnel_numͨ����,kernel_len����˱߳�,image_lenͼ��߳���kernel_num��ǩ����

%�����net��һ���ṹ����layer,result,back��������
%�����layer��һ��cell��������4��cell,����4���㣬������Ĵ�С�ֱ�Ϊ
%10*1*3*3��10*10*3*3+1*10*10,10*10*3*3+2*10*10,10*10*3*3+3*10*10
%�˴�Ĭ��Ϊ�����
%�����result��һ��cell��Ҳ��4��cell������4�������������������С�ֱ�Ϊ
%10*27*27��10*9*9��10*3*3��10
%�˴�Ĭ��Ϊ0
%back�����洢���򴫲��Ľ������resultһ����
%%
function [net]= setparameter(tunnel_num,kernel_len, image_len, kernel_num)
disp('���������У����Ժ�')
%��������ͼ���С�������ǩ��Ŀ����������
%����Ϊͼ��߳��Ծ���˱߳�ȡlog��+1
layer_num= round(log(image_len)/log(kernel_len))+1;
layer=cell(1,layer_num);
result=cell(1,layer_num);
%ÿ��������ĿΪ��ǩ��Ŀ
%��һ�����⴦��һ��
layer{1}=cell(1,tunnel_num);
for i=1:tunnel_num
    layer{1}{i}=rand(kernel_num,kernel_len,kernel_len);
end

%����Ĳ�ͳһ����
for i=2:layer_num
    layer{i}=cell(1,2);
    layer{i}{1}=rand(kernel_num,kernel_num,kernel_len,kernel_len);
    layer{i}{2}=rand(i-1,kernel_num,kernel_num);
end

%���úø������
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