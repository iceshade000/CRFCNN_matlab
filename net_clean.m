%%
%��net��result��back�ĳ�ֵ��Ϊ0

%%
function [net]=net_clean(net)
    layer_num=size(net.result,2);
    result=cell(1,layer_num);
    for i=1:layer_num
        temp=size(net.result{i},2);
        result{i}=zeros(size(net.result{i},1),temp,temp);
    end
    back=result;
    net.back=back;
    net.result=result;
end