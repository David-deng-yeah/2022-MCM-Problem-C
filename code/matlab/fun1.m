%Ŀ�꺯�� ��Ҫ�ĳ�min����ʽ

function f=fun1(w)
%ʾ��
load('gold.mat');
load('bitCoin.mat');

r_gold=[gold.LSTM];%���������� 826 x 1
r_bitcoin=[bitCoin.LSTM];%���������� 826 x 1

zero=rand([length(r_gold),1])*0.0001; % 826 x 1
covlist = [zero r_gold r_bitcoin]; %826 x 3

mu=[0;mean(r_gold);mean(r_bitcoin)];%3*1 ��ֵ����
alpha_gold = 0.01;%�ƽ������
alpha_b = 0.02;
price_gold=1788.80209960938;%�ƽ�ļ۸�
price_b = 43730.0796875000;
gamma =1;%�������ϵ��
tc = [0; - alpha_gold * price_gold; - alpha_b * price_b;];%3*1
sigma = cov(covlist);    
     
maxf=mu'*(w + tc) - gamma/2 * (w + tc)' * sigma * (w + tc);
f = -maxf;%min����ʽ