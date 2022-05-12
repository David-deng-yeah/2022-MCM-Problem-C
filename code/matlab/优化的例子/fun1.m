%目标函数 需要改成min的形式

function f=fun1(w)
global i;
%示例
load('gold.mat');
load('bitCoin.mat');
%%%%%%%%%%%%%%%%%%%参数
r_gold=[gold.farch];%收益率序列
r_bitcoin=[bitCoin.farch];%收益率序列

r_gold=r_gold(1:26+i);
r_bitcoin=r_bitcoin(1:26+i);

zero=rand([length(r_gold),1])*0.0001;
covlist = [zero r_gold r_bitcoin];

mu=[0;mean(r_gold);mean(r_bitcoin)];%3*1 均值向量
alpha_gold = 0.01;%黄金的手续
alpha_b = 0.02;
price_gold=1788.80209960938;%黄金的价格
price_b = 43730.0796875000;
gamma =1;%风险厌恶系数
tc = [0; - alpha_gold * price_gold; - alpha_b * price_b;];%3*1
sigma = cov(covlist);

     
maxf=mu'*(w + tc) - gamma/2 * (w + tc)' * sigma * (w + tc);
f = -maxf;%min的形式