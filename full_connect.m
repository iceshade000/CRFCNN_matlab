%%

%%
function [result]=full_connect(output,input,lambda)
    temp=maxpooling(input,zeros(size(input,1),size(output,2),size(output,3)));%10*1
    temp=lambda*temp;
    %Ȼ��ͨ��crf��
    result=crfprob(temp);
end