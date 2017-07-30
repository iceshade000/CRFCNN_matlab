%%
%输入：网络net，输入标签input_label,步长mu
%输出：反向传播更新完权值的net
%%
function [net]=backward(net,input_label,input_image,mu)
    layer_num=size(net.result,2);
    %卷积核大小
    kernel=size(net.layer{1},3);
    if size(input_label,2)~=1
        input_label=input_label';
    end
    %最后一层的残差
    net.back{layer_num}=input_label-net.result{layer_num};   
    
    
    delta=net.back{layer_num};
    delta=change(delta,net.result{layer_num});
    net.back{layer_num-1}=net.back{layer_num-1}+back_fc(net.result{layer_num-1},delta,net.layer{layer_num});
    
    for i=2:layer_num-1
        now=layer_num-i+1;
        %第t层的误差反向传播到前面的层

        noww=net.result{now};
        delta=net.back{now};
        %利用概率分布改变残差权重
        delta=change(delta,noww);
        
        %卷积部分              
        temp_back=back_conv(net.result{now-1},noww,delta,net.layer{now});
        net.back{now-1}=net.back{now-1}+temp_back;
    end
    %%
    %利用反向传播的误差，对权值进行更新
    for i=1:layer_num
        %从第4层的权值开始更新,一直更新到第二层
        now=layer_num-i+1;
        
        noww=net.result{now};
        delta=net.back{now};
        %利用概率分布改变残差权重
        delta=change(delta,noww);
        %============================================================================
        %卷积层处理
        
        %使用maxpooling增大卷积核的感受野
        if now>1
            input_blob=maxpooling(net.result{now-1},kernel);%10*3*3
        else
            input_blob=input_image;
        end
        net.layer{now}=weight_conv(input_blob,delta,net.layer{now},mu);    
        
        %对偏置进行处理
        for k_now=1:size(net.bias{now},1)
            for k_pre=1:size(net.bias{now},2)
                net.bias{now}(k_now,k_pre)=net.bias{now}(k_now,k_pre)+mu*sum(sum(delta(k_now,:,:)));
            end
        end
    end    
end

function [delta]=change(delta,noww)
        for dk=1:size(delta,1)
        %从现在的第dk个卷积核开始,每个占比修改一下
            for di=1:size(delta,2)
                for dj=1:size(delta,3)
                    delta(dk,di,dj)=delta(dk,di,dj)*noww(dk,di,dj)*(1-noww(dk,di,dj));
                end
            end
        end
end