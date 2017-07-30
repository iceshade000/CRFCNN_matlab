%%
%测试函数
%输入训练好的网络以及测试集，进行测试
%%
function []=testnet(net,test_image,test_label)
    %获取图像数目
    image_num=size(test_label,1);
    
    right=0;
    
    for i=1:image_num
        input_image=test_image{1}(i,:,:);
        input_label=test_label(i,:);
        net=forward(net,input_image);
        [ma,I]=max(squeeze(net.result{size(net.back,2)}));
        if input_label(I)==1
            right=right+1;
        end
        if mod(i,200)==0
            disp([num2str(i),'  ',num2str(right)]);
        end
    end
    disp('正确率为');
    disp(right);
    %前向计算，得出结果
end