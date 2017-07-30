%%
%乘加层，前面层maxpooling变为输入，输入和对应参数相乘，然后再相加,再变成概率
%输入：网络net,输入图像input_image，第几层layer_i
%输出：resulti：第i层的输出
%%
function [resulti] = addtime(net,input_image,layer_i)
    result_now = net.result{layer_i};   
    layer=net.layer{layer_i};
    
    bias=net.bias{layer_i};
    %求出卷积核的边长
    kernel=size(net.layer{1},3);
    image_len=size(input_image,2);
    if layer_i == 1
        for i=1:size(layer,1)
            for j=1:size(layer,2)
                %第i个卷积核，第j个通道
                input_temp=squeeze(input_image(j,:,:));
                result_now(i,:,:) = imfilter(input_temp,squeeze(layer(i,j,:,:)));
                result_now(i,:,:)=result_now(i,:,:)+bias(i,j)*ones(1,image_len,image_len);
            end
        end   
    else
        %后面的层分为两个部分，第一部分与第一层类似
        %使用maxpooling增大卷积核的感受野
        result_pre=net.result{layer_i-1};
        input_blob=maxpooling(result_pre,kernel);%10*9*9
                
        [m,n,n1]=size(input_blob);
        
        for k_now=1:m
            %对现在的每个卷积核
            for k_pre=1:m
                %对之前的每个卷积核
                temp_filter=squeeze(layer(k_now,k_pre,:,:));%3*3
                temp_input=squeeze(input_blob(k_pre,:,:));%9*9
                temp=imfilter(temp_input,temp_filter);
                tempnum=size(temp,1);
                result_now(k_now,:,:)=result_now(k_now,:,:)+reshape(temp(:,:),1,tempnum,tempnum);
                result_now(k_now,:,:)=result_now(k_now,:,:)+bias(k_now,k_pre)*ones(1,tempnum,tempnum);
            end
        end
    
    end
    
    resulti=crfprob(result_now);   
end
