%%
%���룺����net�������ǩinput_label,����mu
%��������򴫲�������Ȩֵ��net
%%
function [net]=backward(net,input_label,input_image,mu)
    layer_num=size(net.result,2);
    %����˴�С
    kernel=size(net.layer{1},3);
    if size(input_label,2)~=1
        input_label=input_label';
    end
    %���һ��Ĳв�
    net.back{layer_num}=input_label-net.result{layer_num};   
    
    
    delta=net.back{layer_num};
    delta=change(delta,net.result{layer_num});
    net.back{layer_num-1}=net.back{layer_num-1}+back_fc(net.result{layer_num-1},delta,net.layer{layer_num});
    
    for i=2:layer_num-1
        now=layer_num-i+1;
        %��t������򴫲���ǰ��Ĳ�

        noww=net.result{now};
        delta=net.back{now};
        %���ø��ʷֲ��ı�в�Ȩ��
        delta=change(delta,noww);
        
        %�������              
        temp_back=back_conv(net.result{now-1},noww,delta,net.layer{now});
        net.back{now-1}=net.back{now-1}+temp_back;
    end
    %%
    %���÷��򴫲�������Ȩֵ���и���
    for i=1:layer_num
        %�ӵ�4���Ȩֵ��ʼ����,һֱ���µ��ڶ���
        now=layer_num-i+1;
        
        noww=net.result{now};
        delta=net.back{now};
        %���ø��ʷֲ��ı�в�Ȩ��
        delta=change(delta,noww);
        %============================================================================
        %����㴦��
        
        %ʹ��maxpooling�������˵ĸ���Ұ
        if now>1
            input_blob=maxpooling(net.result{now-1},kernel);%10*3*3
        else
            input_blob=input_image;
        end
        net.layer{now}=weight_conv(input_blob,delta,net.layer{now},mu);    
        
        %��ƫ�ý��д���
        for k_now=1:size(net.bias{now},1)
            for k_pre=1:size(net.bias{now},2)
                net.bias{now}(k_now,k_pre)=net.bias{now}(k_now,k_pre)+mu*sum(sum(delta(k_now,:,:)));
            end
        end
    end    
end

function [delta]=change(delta,noww)
        for dk=1:size(delta,1)
        %�����ڵĵ�dk������˿�ʼ,ÿ��ռ���޸�һ��
            for di=1:size(delta,2)
                for dj=1:size(delta,3)
                    delta(dk,di,dj)=delta(dk,di,dj)*noww(dk,di,dj)*(1-noww(dk,di,dj));
                end
            end
        end
end