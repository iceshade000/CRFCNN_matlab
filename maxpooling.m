%%
%maxpooling�㣬���˼Ӳ�ʹ�ã�֮ǰ����������maxpooling��Ϊ֮��������
%���룺blobǰһ������,kernel����˵Ĵ�С
%�����maxpooling�Ľ��temp

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