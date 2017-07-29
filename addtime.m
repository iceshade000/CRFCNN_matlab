%%
%�˼Ӳ㣬ǰ���maxpooling��Ϊ���룬����Ͷ�Ӧ������ˣ�Ȼ�������,�ٱ�ɸ���
%���룺����net,����ͼ��input_image���ڼ���layer_i
%�����resulti����i������
%%
function [resulti] = addtime(net,input_image,layer_i)
    result_now = net.result{layer_i};
    result_pre=net.result;
    layer=net.layer{layer_i};
    %�������˵ı߳�
    kernel=size(net.layer{1}{1},2);
    if layer_i == 1
        for i=1:size(layer{1},1)
            result_now(i,:,:) = imfilter(input_image,squeeze(layer{1}(i,:,:)));
        end
        
    else
        %����Ĳ��Ϊ�������֣���һ�������һ������
        %ʹ��maxpooling�������˵ĸ���Ұ
        input_blob=maxpooling(result_pre{layer_i-1},kernel);%10*9*9
                
        [m,n,n1]=size(input_blob);
        
        for k_now=1:m
            %�����ڵ�ÿ�������
            for k_pre=1:m
                %��֮ǰ��ÿ�������
                temp_filter=squeeze(layer{1}(k_now,k_pre,:,:));%3*3
                temp_input=squeeze(input_blob(k_pre,:,:));%9*9
                temp=imfilter(temp_input,temp_filter);
                tempnum=size(temp,1);
                result_now(k_now,:,:)=result_now(k_now,:,:)+reshape(temp(:,:),1,tempnum,tempnum);
            end
        end
        
        %10*10*9 10*9*81
        %����reshape��Ľ��������˳���ǣ�ÿһ����90��Ԫ�أ��ֳ�9��
        %ÿ��10������ʾ10������˵ĵ�һ�������ڶ�������
        %��cnn��ͬ����û�ж�ÿ����Ȩֵ����
        %result_now=layer{1}*reshape(input_temp,kernel*kernel*m,n);
        
        %�ټ���ǰ�漸��������maxpooling�Ľ��
        [n1,n2,n3]=size(layer{2});%1*10*10
        tempkernel=1;
        for i=1:n1
            %��ǰ��i��Ľ������maxpooling
            tempkernel=tempkernel*kernel;
            temp_blob=maxpooling(result_pre{layer_i-i},tempkernel);%10*9*9
            for k_now=1:m
                for k_pre=1:m
                    temp=layer{2}(i,k_now,k_pre)*temp_blob(k_pre,:,:);%1*9*9
                    result_now(k_now,:,:) = result_now(k_now,:,:) + temp;
                end
            end
            %����reshapeû�ı��С,û��Σ��
        end     
    end
    
    resulti=crfprob(result_now);   
end
