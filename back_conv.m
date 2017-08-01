%%
%�ȴ�������Ȼ����maxpooling
%������now����pre������Ϊ
%���룺preǰһ��Ľ������maxpoolingͬʱԼ�������С,now���ڲ�Ľ��
%delta�ǲв�lambda�õ��ľ������
%�����back����ǰһ������
%%
function [back]=back_conv(pre,delta,lambda)
    [ln1,ln2,ln3,ln4]=size(lambda);%20,10��3��3
    %[pn1,pn2,pn3]=size(pre);%10*9*9
    [dn1,dn2,dn3]=size(delta);%20*3*3
    pre_temp=zeros(dn1,dn2,dn3);%ǰһ�����maxpooling��Ĳв�10*3*3
    
    for i=1:ln1     
        temp_input=squeeze(delta(i,:,:));%3*3
        for j=1:ln2
            %���ص�ǰһ���j�����������
            temp_filter=squeeze(lambda(i,j,:,:));
            %���ｫ�������ת180����
            pre_temp(j,:,:)=imfilter(temp_input,rot90(temp_filter,2));
        end   
    end
        
    %���ؾ����Ϻ��ͨ��maxpooling��
    back=back_maxp(pre,pre_temp);
end