%%
%���룺����net�������ǩinput_label,����mu
%��������򴫲�������Ȩֵ��net
%%
function [net]=backward(net,input_label,input_image,mu)
    layer_num=size(net.result,2);
    if size(input_label,2)~=1
        input_label=input_label';
    end
    %���һ��Ĳв�
    net.back{layer_num}=input_label-net.result{layer_num};    
    for i=1:layer_num-1
        now=layer_num-i+1;
        %��t������򴫲���ǰ��Ĳ�
        %�������              
        net.back{now-1}=back_conv(net.result{now-1},net.result{now},net.back{now},net.layer{now}{1});
        
        %maxpooling����
        for j=1:now-1
            lambda=squeeze(net.layer{now}{2}(now-j,:,:));%10*10
            temp=net.back{now},lambda
            net.back{j}=back_maxp(net.result{j},temp);
        end
        
    end
    %���÷��򴫲�������Ȩֵ���и���
    for i=1:layer_num-1
        %�ӵ�4���Ȩֵ��ʼ����,һֱ���µ��ڶ���
        now=layer_num-i+1;
        p=net.result{now};
        delta=net.back{now};
        %net.layer{now}{1}() = net.layer{now}{1}() + mu* delta() * p() *
        %(1-p()) *net.result{now-1}
    end
    
    %���µ�һ���Ȩֵ
    len=size(input_image,2);
    reshape(input_image,1,len*len)
end