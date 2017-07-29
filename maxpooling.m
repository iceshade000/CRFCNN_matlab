%%
%maxpooling层，被乘加层使用，之前层的输出经过maxpooling作为之后层的输入
%输入：blob前一层的输出,kernel卷积核的大小
%输出：maxpooling的结果temp

%%
function [temp]=maxpooling(blob,kernel)
    [m,n,n1] = size(blob);%10,27,27
    new_len=floor(n/kernel);%9
    temp= zeros(m,new_len,new_len);
  for iter=1:m
      newblob=squeeze(blob(iter,:,:));
    for p=1:new_len
        for q=1:new_len
            temp(iter,p,q)=max(max(newblob(p*kernel-kernel+1:p*kernel,q*kernel-kernel+1:q*kernel)));
        end
    end
  end
end