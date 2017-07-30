%%

%%
function [back]=back_fc(result,delta,lambda)
    temp=lambda'*delta;
    back=back_maxp(result,temp);
end