%%

%%
function [result]=detectnet(net,input_image,input_label)
    net=forward(net,input_image);
    layer_num=size(net.result,2);

    if size(input_label,2)~=1
        input_label=input_label';
    end
    %���һ��Ĳв�,�ṩһ������
    net.back{layer_num}=input_label;   
    
    
    delta=net.back{layer_num};
    net.back{layer_num-1}=net.back{layer_num-1}+back_fc(net.result{layer_num-1},delta,net.layer{layer_num});
    
    for i=2:layer_num-1
        now=layer_num-i+1;
        %��t������򴫲���ǰ��Ĳ�

        delta=net.back{now};
        
        %�������              
        temp_back=back_conv(net.result{now-1},delta,net.layer{now});
        net.back{now-1}=net.back{now-1}+temp_back;
    end

    lambda=net.layer{1};
    delta=net.back{1};
    [ln1,ln2,ln3,ln4]=size(lambda);%10,1��3��3
    [dn1,dn2,dn3]=size(delta);%10*3*3
    pre_temp=zeros(ln2,dn2,dn3);%ǰһ�����maxpooling��Ĳв�10*3*3
    
    for i=1:ln1     
        temp_input=squeeze(delta(i,:,:));%3*3
        for j=1:ln2
            %���ص�ǰһ���j�����������
            temp_filter=squeeze(lambda(i,j,:,:));
            %���ｫ�������ת180����
            pre_temp(j,:,:)=imfilter(temp_input,rot90(temp_filter,2));
        end   
    end
    result=squeeze(pre_temp);
    
end