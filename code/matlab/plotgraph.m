load('sense.mat');
data = sense;

hb=bar3(data);  %%  ������ά��״ͼ<br>
for j=1:length(hb)    %% �������Ӹ߶�������ɫ
    zdata=get(hb(j),'Zdata');
    set(hb(j),'Cdata',zdata)
end
 
colormap(YlOrBr);  %% �ı�colormap��ɫ
colorbar     %% �����ɫ��
set(gca,'XTickLabel',{'0.1',' ','0.3',' ','0.5',' ','0.7',' ','0.9'},...
    'yticklabel',{'0.1',' ','0.3',' ','0.5',' ','0.7',' ','0.9'})    %% ���� x���y��̶ȱ�ǩ
xlabel('x');ylabel('y');zlabel('z')  %% ����x y z ���ǩ
box off  %% ȥ���߿�
grid on   %% ��������