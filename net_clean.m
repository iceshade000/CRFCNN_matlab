%%
%��net��result��back�ĳ�ֵ��Ϊ0

%%
function [net]=net_clean(net)
    layer_num=size(net.result,2);
    result=cell(1,layer_num);
    kernel_num=size(net.result{1},1);
    for i=1:layer_num
        temp=size(net.result{i},2);
        result{i}=zeros(kernel_num,temp,temp);
    end
    back=result;
    net.back=back;
    net.result=result;
end