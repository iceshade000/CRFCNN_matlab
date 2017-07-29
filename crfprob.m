%%
%crf概率层，获取从乘加层得到的结果，计算出crf后验概率并归一化
%输入：blob乘加的结果
%输出：result，进行crf概率归一化的结果
%%
function [result]=crfprob(blob)
    [n1,n2,n3]=size(blob);%10*9*9
    temp=exp(blob);
    for i=1:n2
        for j=1:n3
            all=0;
            for t=1:n1
              all=all+temp(t,i,j);
            end
            for t=1:n1
                temp(t,i,j)=temp(t,i,j)/all;
            end
        end
    end
    result=temp;
end