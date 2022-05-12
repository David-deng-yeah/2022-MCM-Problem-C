load('sense.mat');
data = sense;

hb=bar3(data);  %%  绘制三维柱状图<br>
for j=1:length(hb)    %% 根据柱子高度设置颜色
    zdata=get(hb(j),'Zdata');
    set(hb(j),'Cdata',zdata)
end
 
colormap(YlOrBr);  %% 改变colormap颜色
colorbar     %% 添加颜色条
set(gca,'XTickLabel',{'0.1',' ','0.3',' ','0.5',' ','0.7',' ','0.9'},...
    'yticklabel',{'0.1',' ','0.3',' ','0.5',' ','0.7',' ','0.9'})    %% 设置 x轴和y轴刻度标签
xlabel('x');ylabel('y');zlabel('z')  %% 设置x y z 轴标签
box off  %% 去掉边框
grid on   %% 保留网格