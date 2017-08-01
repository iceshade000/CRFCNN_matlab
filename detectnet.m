%%

%%
function [result]=detectnet(net,input_image,input_label)
    net=forward(net,input_image);
    layer_num=size(net.result,2);

    if size(input_label,2)~=1
        input_label=input_label';
    end
    %最后一层的残差,提供一个激活
    net.back{layer_num}=input_label;   
    
    
    delta=net.back{layer_num};
    net.back{layer_num-1}=net.back{layer_num-1}+back_fc(net.result{layer_num-1},delta,net.layer{layer_num});
    
    for i=2:layer_num-1
        now=layer_num-i+1;
        %第t层的误差反向传播到前面的层

        delta=net.back{now};
        
        %卷积部分              
        temp_back=back_conv(net.result{now-1},delta,net.layer{now});
        net.back{now-1}=net.back{now-1}+temp_back;
    end

    lambda=net.layer{1};
    delta=net.back{1};
    [ln1,ln2,ln3,ln4]=size(lambda);%10,1，3，3
    [dn1,dn2,dn3]=size(delta);%10*3*3
    pre_temp=zeros(ln2,dn2,dn3);%前一层输出maxpooling后的残差10*3*3
    
    for i=1:ln1     
        temp_input=squeeze(delta(i,:,:));%3*3
        for j=1:ln2
            %返回到前一层第j个卷积层的输出
            temp_filter=squeeze(lambda(i,j,:,:));
            %这里将卷积核旋转180即可
            pre_temp(j,:,:)=imfilter(temp_input,rot90(temp_filter,2));
        end   
    end
    result=squeeze(pre_temp);
    
end