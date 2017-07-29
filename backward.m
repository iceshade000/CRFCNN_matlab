%%
%输入：网络net，输入标签input_label,步长mu
%输出：反向传播更新完权值的net
%%
function [net]=backward(net,input_label,input_image,mu)
    layer_num=size(net.result,2);
    kernel=size(net.layer{1}{1},2);
    if size(input_label,2)~=1
        input_label=input_label';
    end
    %最后一层的残差
    net.back{layer_num}=input_label-net.result{layer_num};    
    for i=1:layer_num-1
        now=layer_num-i+1;
        %第t层的误差反向传播到前面的层

        noww=net.result{now};
        delta=net.back{now};
        %利用概率分布改变残差权重
        for dk=1:size(delta,1)
        %从现在的第dk个卷积核开始,每个占比修改一下
            for di=1:size(delta,2)
                for dj=1:size(delta,3)
                    delta(dk,di,dj)=delta(dk,di,dj)*noww(dk,di,dj)*(1-noww(dk,di,dj));
                end
            end
        end
        
        %卷积部分              
        temp_back=back_conv(net.result{now-1},noww,delta,net.layer{now}{1});
        net.back{now-1}=net.back{now-1}+temp_back;
        
        %maxpooling部分
        for j=1:now-1
            %选好参数
            lambda=squeeze(net.layer{now}{2}(now-j,:,:));%10*10
            temp=zeros(size(delta,1),size(delta,2),size(delta,3));
            %返回到乘加前
            for now_i=1:size(lambda,1)
                for pre_i=1:size(lambda,2)
                    temp(pre_i,:,:)=temp(pre_i,:,:)+lambda(now_i,pre_i)*delta(now_i,:,:);                 
                end
            end
            %返回到maxpooling处
            net.back{j}=net.back{j}+back_maxp(net.result{j},temp);
        end       
    end
    %%
    %利用反向传播的误差，对权值进行更新
    for i=1:layer_num-1
        %从第4层的权值开始更新,一直更新到第二层
        now=layer_num-i+1;
        
        noww=net.result{now};
        delta=net.back{now};
        %利用概率分布改变残差权重
        for dk=1:size(delta,1)
        %从现在的第dk个卷积核开始,每个占比修改一下
            for di=1:size(delta,2)
                for dj=1:size(delta,3)
                    delta(dk,di,dj)=delta(dk,di,dj)*noww(dk,di,dj)*(1-noww(dk,di,dj));
                end
            end
        end
        %============================================================================
        %卷积层处理
        
        %使用maxpooling增大卷积核的感受野
        input_blob=maxpooling(net.result{now-1},kernel);%10*3*3
        net.layer{now}{1}=weight_conv(input_blob,delta,net.layer{now}{1},mu);
        
        %===================================================================
        %maxpooling层处理
        m=size(net.layer{1}{1},1);
        kernel=size(net.layer{1}{1},2);
        tempkernel=1;
        for pre_i=1:now-1
            %往前数pre_i层的结果进行maxpooling
            tempkernel=tempkernel*kernel;
            temp_blob=maxpooling(net.result{now-pre_i},tempkernel);%10*9*9
            for k_now=1:m
                for k_pre=1:m
                    temp=squeeze(temp_blob(k_pre,:,:)).*squeeze(delta(k_now,:,:));
                    net.layer{now}{2}(pre_i,k_now,k_pre)=net.layer{now}{2}(pre_i,k_now,k_pre)+mu*sum(sum(temp));
                end
            end
        end     
       
        
        
    end
    
    %============================================================================
    %更新第一层的权值
    noww=net.result{1};
    delta=net.back{1};
        %利用概率分布改变残差权重
    for dk=1:size(delta,1)
        %从现在的第dk个卷积核开始,每个占比修改一下
        for di=1:size(delta,2)
            for dj=1:size(delta,3)
                delta(dk,di,dj)=delta(dk,di,dj)*noww(dk,di,dj)*(1-noww(dk,di,dj));
            end
        end
    end
    %net.layer{1}{1}=weight_conv(input_image,delta,net.layer{1}{1},mu);
    lambda=net.layer{1}{1};
    [dn1,dn2,dn3]=size(delta);%10*27*27
    [ln1,ln2,ln3]=size(lambda);%10,3，3
    kernel=ln3;
    gap=floor((kernel-1)/2);

    %先将输入拓展一圈
    input_temp=zeros(dn2+kernel-1,dn3+kernel-1);
    input_temp(gap+1:gap+dn2,gap+1:gap+dn3)=input_image(:,:);
        
    for i=1:ln1
        %第i个输出图，目标是求出3*3的卷积和            
        for a=1:ln2
            for b=1:ln3
                temp=input_temp(a:a+dn2-1,b:b-1+dn3).*squeeze(delta(i,:,:));
                lambda(i,a,b)=lambda(i,a,b)+mu*sum(sum(temp));
            end
        end                 
    end
    net.layer{1}{1}=lambda;

    
    
end