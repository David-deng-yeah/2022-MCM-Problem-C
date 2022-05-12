##  LSTM原理

长短期记忆网络（Long Short Term Memory，简称LSTM）模型， 本质上是一种特殊的循环神经网络（Recurrent Neural Network，简称RNN）。lstm模型在rnn模型的基础上通过增加门限（gate）来解决rnn短期记忆的问题，使得rnn能够有效利用长距离的序列信息。

lstm在rnn的基础上增加了三个逻辑控制单元

* 输入门限（input gate）
* 输出门限（output gate）
* 遗忘门限（forget gate）

这三个逻辑控制单元各自连接到了一个乘法元件上（如下图所示），通过设定神经网络的记忆单元与其他部分的连接权重，从而控制数据流的输入、输出以及细胞单元（memory cell）的状态， 具体概念图如下

![](C:%5CUsers%5C86135%5CDesktop%5C%E8%B5%84%E6%96%99%5C%E6%AF%94%E8%B5%9B%5C%E7%BE%8E%E8%B5%9B%5CLSTM%E5%8E%9F%E7%90%86%E5%9B%BE%5Clstm%E6%A6%82%E5%BF%B5%E5%9B%BE.jpg)

上图中具体部件的描述如下：

* input  gate ： 控制信息是否流入，记为$i_t$
* forget gate : 控制上一时刻的memory cell的信息是否积累到当前时刻的memory cell中， 记为$f_t$
* output gate : 控制当前时刻的memory cell的信息是否流入当前的隐藏状态$h_t$中， 记为$o_t$
* cell : 记忆单元， 表示神经元状态的记忆， 使得lstm单元具有保存、读取、重置和更新长距离历史信息的能力，记为$c_t$

在$t$时刻，lstm模型的公式定义如下:

![](C:%5CUsers%5C86135%5CDesktop%5C%E8%B5%84%E6%96%99%5C%E6%AF%94%E8%B5%9B%5C%E7%BE%8E%E8%B5%9B%5CLSTM%E5%8E%9F%E7%90%86%E5%9B%BE%5C%E5%85%AC%E5%BC%8F.JPG)

根据公式描述，lstm的细节图如下。在lstm神经网络的训练过程中，首先将t时刻的数据特征输入至输入层，经过激励函数输出结果； 然后将输出结果、$t-1$时刻的隐藏层输出和$t-1$时刻cell单元存储的信息输入lstm结构的节点中，通过input gate、output gate、forget gate和cell单元的处理，输出数据到下一隐藏层或输出层，输出lstm结构节点的结果到输出层神经元，计算反向传播误差，并更新各个权值。

lstm细节图如下

![](C:%5CUsers%5C86135%5CDesktop%5C%E8%B5%84%E6%96%99%5C%E6%AF%94%E8%B5%9B%5C%E7%BE%8E%E8%B5%9B%5CLSTM%E5%8E%9F%E7%90%86%E5%9B%BE%5Clstm%E7%BB%86%E8%8A%82%E5%9B%BE.jpg)

##  数据处理及特征工程（以黄金为例）

1. 原始的数据表如下（gold_select.csv），1826个样本，16个列

![image-20220221123655767](C:%5CUsers%5C86135%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220221123655767.png)

去掉date之后如下（1826x15）其中 USD（PM）为预测目标，**剩下的14个作为特征**

![image-20220221123723261](C:%5CUsers%5C86135%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220221123723261.png)

本次任务中，我们利用lstm将**时间序列预测问题**转化为了**监督学习问题**

我们知道，时间序列预测的本质主要是根据前T个时刻的观测数据（特征）来预测处第T+1个时刻的时间序列的值，这就可以转化为机器学习中的监督学习问题了，即利用之前的样本训练出一个预测模型，对新的输入样本进行预测，具体原理如下

我们根据超参数 n_in（滞后期数），截取前 n_in 个时刻的**所有列（包括预测目标）**，作为第 t 个时刻的特征，比如本次任务，我们的 n_in = 1，所以我们提取前 1个 时刻的特征，作为第 t 个时刻的特征，最后提取的数据表如下

如下图所示，第t个时刻的样本具有16列，其中最后一列 Y(t)为第t个时刻样本的**预测目标**， 前面15列为**前一个样本的所有列**

![image-20220221123837567](C:%5CUsers%5C86135%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220221123837567.png)

当然，我们的模型可以预测未来多个时刻的值， 我们也能提取多个前面时刻的所有列作为特征， 具体的公式如下

n_vars * n_in + n_out

加入我们要预测未来一步，且提取前一个时刻的所有列，且我们有（1个预测目标+14个特征 = 15）列的数据表

计算过程为 15x1+1 = 16

 我们本次的数据为 1825x16， 要转换为（样本个数，滞后期数，特征个数）的维度来作为lstm的输入

于是数据表（1825x16）转化为 （1825，1，15）的data

data再按照设定（前999天观望那个，第一千天投资）为训练集（999，1，15）和测试集（826，1，15）

**ok，最终的数据为：**

* **train_X (999,1,15)**
* **test_X (826, 1, 15)**

##  模型的结构

**(还会更新)**

本次的神经网络结构为 120个隐藏层节点的lstm层， 1个全连接层的神经网络

![image-20220221125803481](C:%5CUsers%5C86135%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220221125803481.png)

##  模型的超参数

* n_in : 滞后期数
* n_out : 超前预测数
* n_vars : 数据表的特征个数
* n_neuron : lstm的隐藏层神经元个数
* n_batch : 批次大小，也就是一次训练选取的样本个数
* n_epoch : 模型在整个训练数据集中的工作次数
* repeats : 训练的模型个数（我们采取训练国歌模型取平均的方法，增加模型的稳定性）

![image-20220221125942756](C:%5CUsers%5C86135%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20220221125942756.png)

##  模型的训练

采用 n_batch的批次大小，n_epoch的工作次数，训练 repeats个模型，模型之间求均值和std，下图为模型的loss值下降过程，虚色为置信区间，实线为下降趋势

![](C:%5CUsers%5C86135%5CDesktop%5C%E8%B5%84%E6%96%99%5C%E6%AF%94%E8%B5%9B%5C%E7%BE%8E%E8%B5%9B%5Closs%E4%B8%8B%E9%99%8D%E8%B6%8B%E5%8A%BF.png)

##  拟合效果

下图为模型对测试集的拟合效果

![](C:%5CUsers%5C86135%5CDesktop%5C%E8%B5%84%E6%96%99%5C%E6%AF%94%E8%B5%9B%5C%E7%BE%8E%E8%B5%9B%5Clstm%E6%B5%8B%E8%AF%95%E9%9B%86%E6%8B%9F%E5%90%88%E6%95%88%E6%9E%9C_%E9%BB%84%E9%87%91.png)



##  