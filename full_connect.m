%%

%%
function [result]=full_connect(input,lambda)
    temp=maxpooling(input,size(input,2));%10*1
    temp=lambda*temp;
    %Ȼ��ͨ��crf��
    result=crfprob(temp);
end