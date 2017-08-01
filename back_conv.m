%%
%先处理卷积，然后处理maxpooling
%将误差从now传向pre，参数为
%输入：pre前一层的结果用于maxpooling同时约束输出大小,now现在层的结果
%delta是残差lambda用到的卷积参数
%输出：back传到前一层的误差
%%
function [back]=back_conv(pre,delta,lambda)
    [ln1,ln2,ln3,ln4]=size(lambda);%20,10，3，3
    %[pn1,pn2,pn3]=size(pre);%10*9*9
    [dn1,dn2,dn3]=size(delta);%20*3*3
    pre_temp=zeros(dn1,dn2,dn3);%前一层输出maxpooling后的残差10*3*3
    
    for i=1:ln1     
        temp_input=squeeze(delta(i,:,:));%3*3
        for j=1:ln2
            %返回到前一层第j个卷积层的输出
            temp_filter=squeeze(lambda(i,j,:,:));
            %这里将卷积核旋转180即可
            pre_temp(j,:,:)=imfilter(temp_input,rot90(temp_filter,2));
        end   
    end
        
    %反回卷积完毕后就通过maxpooling层
    back=back_maxp(pre,pre_temp);
end