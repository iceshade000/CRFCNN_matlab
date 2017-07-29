%%
%乘加层，前面层maxpooling变为输入，输入和对应参数相乘，然后再相加,再变成概率
%输入：网络net,输入图像input_image，第几层layer_i
%输出：resulti：第i层的输出
%%
function [resulti] = addtime(net,input_image,layer_i)
    result_now = net.result{layer_i};
    result_pre=net.result;
    layer=net.layer{layer_i};
    %求出卷积核的边长
    kernel=size(net.layer{1}{1},2);
    if layer_i == 1
        for i=1:size(layer{1},1)
            result_now(i,:,:) = imfilter(input_image,squeeze(layer{1}(i,:,:)));
        end
        
    else
        %后面的层分为两个部分，第一部分与第一层类似
        %使用maxpooling增大卷积核的感受野
        input_blob=maxpooling(result_pre{layer_i-1},kernel);%10*9*9
                
        [m,n,n1]=size(input_blob);
        
        for k_now=1:m
            %对现在的每个卷积核
            for k_pre=1:m
                %对之前的每个卷积核
                temp_filter=squeeze(layer{1}(k_now,k_pre,:,:));%3*3
                temp_input=squeeze(input_blob(k_pre,:,:));%9*9
                temp=imfilter(temp_input,temp_filter);
                tempnum=size(temp,1);
                result_now(k_now,:,:)=result_now(k_now,:,:)+reshape(temp(:,:),1,tempnum,tempnum);
            end
        end
        
        %10*10*9 10*9*81
        %这里reshape后的结果的排列顺序是，每一列有90个元素，分成9份
        %每份10个，表示10个卷积核的第一个数，第二个……
        %与cnn不同，并没有对每个都权值共享
        %result_now=layer{1}*reshape(input_temp,kernel*kernel*m,n);
        
        %再加上前面几层的输出的maxpooling的结果
        [n1,n2,n3]=size(layer{2});%1*10*10
        tempkernel=1;
        for i=1:n1
            %往前数i层的结果进行maxpooling
            tempkernel=tempkernel*kernel;
            temp_blob=maxpooling(result_pre{layer_i-i},tempkernel);%10*9*9
            for k_now=1:m
                for k_pre=1:m
                    temp=layer{2}(i,k_now,k_pre)*temp_blob(k_pre,:,:);%1*9*9
                    result_now(k_now,:,:) = result_now(k_now,:,:) + temp;
                end
            end
            %这里reshape没改变大小,没有危险
        end     
    end
    
    resulti=crfprob(result_now);   
end
