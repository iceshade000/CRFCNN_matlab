%%
%���룺����net�������ǩinput_label,����mu
%��������򴫲�������Ȩֵ��net
%%
function [net]=backward(net,input_label,input_image,mu)
    layer_num=size(net.result,2);
    kernel=size(net.layer{1}{1},2);
    if size(input_label,2)~=1
        input_label=input_label';
    end
    %���һ��Ĳв�
    net.back{layer_num}=input_label-net.result{layer_num};    
    for i=1:layer_num-1
        now=layer_num-i+1;
        %��t������򴫲���ǰ��Ĳ�

        noww=net.result{now};
        delta=net.back{now};
        %���ø��ʷֲ��ı�в�Ȩ��
        for dk=1:size(delta,1)
        %�����ڵĵ�dk������˿�ʼ,ÿ��ռ���޸�һ��
            for di=1:size(delta,2)
                for dj=1:size(delta,3)
                    delta(dk,di,dj)=delta(dk,di,dj)*noww(dk,di,dj)*(1-noww(dk,di,dj));
                end
            end
        end
        
        %�������              
        temp_back=back_conv(net.result{now-1},noww,delta,net.layer{now}{1});
        net.back{now-1}=net.back{now-1}+temp_back;
        
        %maxpooling����
        for j=1:now-1
            %ѡ�ò���
            lambda=squeeze(net.layer{now}{2}(now-j,:,:));%10*10
            temp=zeros(size(delta,1),size(delta,2),size(delta,3));
            %���ص��˼�ǰ
            for now_i=1:size(lambda,1)
                for pre_i=1:size(lambda,2)
                    temp(pre_i,:,:)=temp(pre_i,:,:)+lambda(now_i,pre_i)*delta(now_i,:,:);                 
                end
            end
            %���ص�maxpooling��
            net.back{j}=net.back{j}+back_maxp(net.result{j},temp);
        end       
    end
    %%
    %���÷��򴫲�������Ȩֵ���и���
    for i=1:layer_num-1
        %�ӵ�4���Ȩֵ��ʼ����,һֱ���µ��ڶ���
        now=layer_num-i+1;
        
        noww=net.result{now};
        delta=net.back{now};
        %���ø��ʷֲ��ı�в�Ȩ��
        for dk=1:size(delta,1)
        %�����ڵĵ�dk������˿�ʼ,ÿ��ռ���޸�һ��
            for di=1:size(delta,2)
                for dj=1:size(delta,3)
                    delta(dk,di,dj)=delta(dk,di,dj)*noww(dk,di,dj)*(1-noww(dk,di,dj));
                end
            end
        end
        %============================================================================
        %����㴦��
        
        %ʹ��maxpooling�������˵ĸ���Ұ
        input_blob=maxpooling(net.result{now-1},kernel);%10*3*3
        net.layer{now}{1}=weight_conv(input_blob,delta,net.layer{now}{1},mu);
        
        %===================================================================
        %maxpooling�㴦��
        m=size(net.layer{1}{1},1);
        kernel=size(net.layer{1}{1},2);
        tempkernel=1;
        for pre_i=1:now-1
            %��ǰ��pre_i��Ľ������maxpooling
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
    %���µ�һ���Ȩֵ
    noww=net.result{1};
    delta=net.back{1};
        %���ø��ʷֲ��ı�в�Ȩ��
    for dk=1:size(delta,1)
        %�����ڵĵ�dk������˿�ʼ,ÿ��ռ���޸�һ��
        for di=1:size(delta,2)
            for dj=1:size(delta,3)
                delta(dk,di,dj)=delta(dk,di,dj)*noww(dk,di,dj)*(1-noww(dk,di,dj));
            end
        end
    end
    %net.layer{1}{1}=weight_conv(input_image,delta,net.layer{1}{1},mu);
    lambda=net.layer{1}{1};
    [dn1,dn2,dn3]=size(delta);%10*27*27
    [ln1,ln2,ln3]=size(lambda);%10,3��3
    kernel=ln3;
    gap=floor((kernel-1)/2);

    %�Ƚ�������չһȦ
    input_temp=zeros(dn2+kernel-1,dn3+kernel-1);
    input_temp(gap+1:gap+dn2,gap+1:gap+dn3)=input_image(:,:);
        
    for i=1:ln1
        %��i�����ͼ��Ŀ�������3*3�ľ����            
        for a=1:ln2
            for b=1:ln3
                temp=input_temp(a:a+dn2-1,b:b-1+dn3).*squeeze(delta(i,:,:));
                lambda(i,a,b)=lambda(i,a,b)+mu*sum(sum(temp));
            end
        end                 
    end
    net.layer{1}{1}=lambda;

    
    
end