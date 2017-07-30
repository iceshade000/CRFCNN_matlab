%%
%�˼Ӳ㣬ǰ���maxpooling��Ϊ���룬����Ͷ�Ӧ������ˣ�Ȼ�������,�ٱ�ɸ���
%���룺����net,����ͼ��input_image���ڼ���layer_i
%�����resulti����i������
%%
function [resulti] = addtime(net,input_image,layer_i)
    result_now = net.result{layer_i};   
    layer=net.layer{layer_i};
    
    bias=net.bias{layer_i};
    %�������˵ı߳�
    kernel=size(net.layer{1},3);
    image_len=size(input_image,2);
    if layer_i == 1
        for i=1:size(layer,1)
            for j=1:size(layer,2)
                %��i������ˣ���j��ͨ��
                input_temp=squeeze(input_image(j,:,:));
                result_now(i,:,:) = imfilter(input_temp,squeeze(layer(i,j,:,:)));
                result_now(i,:,:)=result_now(i,:,:)+bias(i,j)*ones(1,image_len,image_len);
            end
        end   
    else
        %����Ĳ��Ϊ�������֣���һ�������һ������
        %ʹ��maxpooling�������˵ĸ���Ұ
        result_pre=net.result{layer_i-1};
        input_blob=maxpooling(result_pre,kernel);%10*9*9
                
        [m,n,n1]=size(input_blob);
        
        for k_now=1:m
            %�����ڵ�ÿ�������
            for k_pre=1:m
                %��֮ǰ��ÿ�������
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
