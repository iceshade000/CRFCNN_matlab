%%
%���Ժ���
%����ѵ���õ������Լ����Լ������в���
%%
function []=testnet(net,test_image,test_label)
    %��ȡͼ����Ŀ
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
    disp('��ȷ��Ϊ');
    disp(right);
    %ǰ����㣬�ó����
end