%%
%输入：网络net，输入图像input_image
%输出：前向计算好的结果result
%%
function [result]=forward(net,input_image)
    image_len=size(input_image,1);
    assert( image_len==size(net.result{1},2),'输入图像大小不对');
    layer_num=size(net.layer,2);
    for i=1:layer_num
        %做好输入矩阵,与卷积层矩阵乘加,
        %乘加的结果变成概率，存到输出net.result{i}中
        net.result{i}=addtime(net,input_image,i);
    end
    result=net.result;
end