%%
%输入：pre前层的结果，now本层的误差
%输出：back传到前一层的误差
%%
function [back]=back_maxp(pre,now)