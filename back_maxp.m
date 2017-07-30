%%
%输入：pre前层的结果，delta是本层的误差
%输出：back传到前一层的误差
%%
function [back]=back_maxp(pre,delta)
    [pn1,pn2,pn3]=size(pre);%10*9*9
    [dn1,dn2,dn3]=size(delta);%10*3*3
    assert( dn2~=0,'delta第二维不为0');
    kernel=floor(pn2/dn2);%3
    back=zeros(pn1,pn2,pn3);
    for i=1:pn1
        %对每个卷积核，找到每个区域中的最大值，然后把误差传过去
        for j=1:dn2
            for k=1:dn3
                [m,n]=maxpoint(squeeze(pre(i,:,:)),j*kernel-kernel+1,j*kernel,k*kernel-kernel+1,k*kernel);
                back(i,m,n)=back(i,m,n)+delta(i,j,k);
            end
        end
    end
end

function [m,n]=maxpoint(input,a,b,c,d)
    m=b;
    n=d;
    for i=a:b
        for j=c:d
            if input(i,j)>input(m,n)
                m=i;
                n=j;
            end
        end
    end
end