%%
%�ȴ�������Ȼ����maxpooling
%������now����pre������Ϊ
%���룺preǰһ��Ľ������maxpoolingͬʱԼ�������С,now���ڲ�Ľ��
%delta�ǲв�lambda�õ��ľ������
%�����back����ǰһ������
%%
function [back]=back_conv(pre,now,delta,lambda)
    [ln1,ln2,ln3]=size(lambda);%10,10��3��3
    [pn1,pn2,pn3]=size(pre);%10*9*9
    [nn1,nn2,nn3]=size(now);%10*3*3
    [dn1,dn2,dn3]=size(delta);%10*3*3
    pre_temp=zeros(dn1,dn2,dn3);%ǰһ�����maxpooling��Ĳв�10*3*3
    
    for i=1:ln1
        %�����ڵĵ�i������˿�ʼ
        for j=1:ln2
            %���ص�ǰһ���j�����������
            temp_input=squeeze(delta(i,:,:));
            temp_filter=squeeze(lambda(i,j,:,:));
            pre_temp(j,:,:)=imfilter(temp_input,temp_filter);
        end
        
    end
    
    p=net.result{now};%10*1
        delta=net.back{now};
        net.back{now-1}() = net.back{now-1}() + delta() * p() * (1-p()) * net.layer{now}{1}();%10*81
    
        
    %���ؾ����Ϻ��ͨ��maxpooling��
    back=back_maxp(pre,pre_temp);

end