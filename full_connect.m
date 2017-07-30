%%

%%
function [result]=full_connect(input,lambda)
    temp=maxpooling(input,size(input,2));%10*1
    temp=lambda*temp;
    %然后通过crf层
    result=crfprob(temp);
end