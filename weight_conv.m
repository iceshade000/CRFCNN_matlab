%%
%���룬input_blob���ǰ��delta�����lambda������mu����
%%
function [lambda]=weight_conv(input_blob,delta,lambda,mu)
    [in1,in2,in3]=size(input_blob);%1*27*27
    [dn1,dn2,dn3]=size(delta);%10*27*27 
    [ln1,ln2,ln3,ln4]=size(lambda);%10,1��3��3

    kernel=ln3;
    gap=floor((kernel-1)/2);
    for j=1:ln2
        %�Ƚ�������չһȦ
        input_temp=zeros(in2+kernel-1,in3+kernel-1);
        input_temp(gap+1:gap+in2,gap+1:gap+in3)=input_blob(j,:,:);
        
        for i=1:ln1
            %��i�����ͼ�͵�j������ͼ��Ŀ�������3*3�ľ����            
            for a=1:ln3
                for b=1:ln4
                    temp=input_temp(a:a+dn2-1,b:b-1+dn3).*squeeze(delta(i,:,:));
                    lambda(i,j,a,b)=lambda(i,j,a,b)+mu*sum(sum(temp));
                end
            end
            
            
        end
    end
end