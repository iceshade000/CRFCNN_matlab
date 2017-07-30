%%
%���룺����net������ͼ��input_image
%�����ǰ�����õĽ��result
%%
function [net]=forward(net,input_image)
    net=net_clean(net);
    image_len=size(input_image,2);
    assert( image_len==size(net.result{1},2),'����ͼ���С����');
    tunnel_num=size(input_image,1);
    assert( tunnel_num==size(net.layer{1},2),'����ͼ��ͨ��������');
    layer_num=size(net.layer,2);
    for i=1:layer_num-1
        %�����������,���������˼�,
        %�˼ӵĽ����ɸ��ʣ��浽���net.result{i}��
        net.result{i}=addtime(net,input_image,i);
    end
    %���һ�����ȫ���Ӳ�
    net.result{layer_num}=full_connect(net.result{layer_num-1},net.layer{layer_num});
    
end