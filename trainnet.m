%%
%ѵ������
%���룺����net,ѵ��ͼ��train_image,ѵ����ǩtrain_label
%%
function [net]=trainnet(net,train_image,train_label)
%��ȡͼ����Ŀ���������Ŀ
[image_num,label_num]=size(train_label);
%����ѵ��1000000��
iternum=1000000;
mu=10;
all=0;
%ѵ������
for i=1:iternum
    %��ͼ�������ѡȡһ��,�Ժ���ԸĽ������ѡȡһ��
    temp=mod(randi(image_num)+30000,image_num)+1;
    input_image=train_image{1}(temp,:,:);
    
    %ǰ����㣬ÿ��1000���������������
    net=forward(net,input_image);
    
    %���򴫲������²�����ֵ
    input_label=train_label(temp,:);
    net=backward(net,input_label,input_image,mu);
    %��������˾�����һ��
    t=net.back{size(net.back,2)};
    if dot(t,t')>0.5 %���������֪��С��0.5�϶�û��
       % net=forward(net,input_image);
        all=all+1;
        %net=backward(net,input_label,input_image,mu);
    end
    
    if mod(i,100)==1;
        disp(['����������',num2str(all)]);
        all=0;
    end
 
    if mod(i,20)==1         
        
        disp(['��',num2str(i),'�ֵ���: ']);
        net=forward(net,input_image);
        [ma,I]=max(squeeze(net.result{size(net.back,2)}));
        disp(['����������',num2str(find(input_label)-1),' ,���������',num2str(I-1)]);
        t2=input_label'-net.result{size(net.result,2)};
        disp(['���Ϊ��',num2str(dot(t,t')),'->',num2str(dot(t2,t2'))]);
        disp(['���ֲ�����',num2str(net.layer{2}(5,5,2,3)),'ƫ��',num2str(net.bias{2}(5,5))]);
    end
    %ÿ��1000�Σ���������
    if mod(i,1000)==0
        disp('��������');
        savenet('net_mnist.mat',net);
    end
end
end