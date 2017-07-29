%%
%���룺����net������ͼ��input_image
%�����ǰ�����õĽ��result
%%
function [result]=forward(net,input_image)
    image_len=size(input_image,1);
    assert( image_len==size(net.result{1},2),'����ͼ���С����');
    layer_num=size(net.layer,2);
    for i=1:layer_num
        %�����������,���������˼�,
        %�˼ӵĽ����ɸ��ʣ��浽���net.result{i}��
        net.result{i}=addtime(net,input_image,i);
    end
    result=net.result;
end