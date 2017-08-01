%%

%%
function [result]=full_connect(output,input,lambda)
    temp=maxpooling(input,zeros(size(input,1),size(output,2),size(output,3)));%10*1
    temp=lambda*temp;
    %然后通过crf层
    result=crfprob(temp);
end