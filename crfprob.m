%%
%crf���ʲ㣬��ȡ�ӳ˼Ӳ�õ��Ľ���������crf������ʲ���һ��
%���룺blob�˼ӵĽ��
%�����result������crf���ʹ�һ���Ľ��
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