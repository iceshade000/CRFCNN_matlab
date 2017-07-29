%%
%输入：网络net，输入标签input_label,步长mu
%输出：反向传播更新完权值的net
%%
function [net]=backward(net,input_label,input_image,mu)
    layer_num=size(net.result,2);
    if size(input_label,2)~=1
        input_label=input_label';
    end
    %最后一层的残差
    net.back{layer_num}=input_label-net.result{layer_num};    
    for i=1:layer_num-1
        now=layer_num-i+1;
        %第t层的误差反向传播到前面的层
        %卷积部分              
        net.back{now-1}=back_conv(net.result{now-1},net.result{now},net.back{now},net.layer{now}{1});
        
        %maxpooling部分
        for j=1:now-1
            lambda=squeeze(net.layer{now}{2}(now-j,:,:));%10*10
            temp=net.back{now},lambda
            net.back{j}=back_maxp(net.result{j},temp);
        end
        
    end
    %利用反向传播的误差，对权值进行更新
    for i=1:layer_num-1
        %从第4层的权值开始更新,一直更新到第二层
        now=layer_num-i+1;
        p=net.result{now};
        delta=net.back{now};
        %net.layer{now}{1}() = net.layer{now}{1}() + mu* delta() * p() *
        %(1-p()) *net.result{now-1}
    end
    
    %更新第一层的权值
    len=size(input_image,2);
    reshape(input_image,1,len*len)
end