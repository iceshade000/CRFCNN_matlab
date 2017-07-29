%%
%输入，input_blob卷积前，delta卷积后，lambda参数，mu步长
%%
function [lambda]=weight_conv(input_blob,delta,lambda,mu)

    [dn1,dn2,dn3]=size(delta);%10*3*3 input_blob和这相同
    [ln1,ln2,ln3,ln4]=size(lambda);%10,10，3，3

    kernel=ln3;
    gap=floor((kernel-1)/2);
    for j=1:ln2
        %先将输入拓展一圈
        input_temp=zeros(dn2+kernel-1,dn3+kernel-1);
        input_temp(gap+1:gap+dn2,gap+1:gap+dn3)=input_blob(j,:,:);
        
        for i=1:ln1
            %第i个输出图和第j个输入图，目标是求出3*3的卷积和            
            for a=1:ln3
                for b=1:ln4
                    temp=input_temp(a:a+dn2-1,b:b-1+dn3).*squeeze(delta(i,:,:));
                    lambda(i,j,a,b)=lambda(i,j,a,b)+mu*sum(sum(temp));
                end
            end
            
            
        end
    end
end