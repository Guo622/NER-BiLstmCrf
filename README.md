# 基于BiLstm+Crf 的命名实体识别（FudanNLP/nlp-beginner task4）

数据集：CONLL 2003，https://www.clips.uantwerpen.be/conll2003/ner/

## 运行

```shell
pip install -r requirements.txt
python train.py
python inference.py
```

## 原理解释

+ https://zhuanlan.zhihu.com/p/97676647

  对下图关于 $\alpha$ 推导进行部分补充（源自上面的链接）：

  ![](https://pic3.zhimg.com/80/v2-2bd7ca4c856a086843b7145b537eaf2e_1440w.jpg)

$$ \begin{aligned} \alpha_{i,j} &= log\Bigg[\sum_{y\in P^{i-1}}exp\bigg( \Psi\big( w_{1\sim i},y_{1\sim i-1},y_i=j\big)\bigg)\Bigg]\\\\&=log\Bigg[\sum_{y\in P^{i-1}}exp\bigg( \Psi\big( w_{1\sim i-1},y_{1\sim i-2},y_{i-1}=j^{\prime}\big)+T_{j^{\prime},j}+E_{i,j}\bigg)\Bigg]\\\\&= log\Bigg[\sum_{y\in P^{i-2}}\sum_{j^{\prime}} exp\bigg( \Psi\big( w_{1\sim i-1},y_{1\sim i-2},y_i-1=j^{\prime}\big)+T_{j^{\prime},j}+E_{i,j}\bigg)\Bigg]\\\\&=log\Bigg[\sum_{j^{\prime}}\sum_{y\in P^{i-2}} exp\bigg( \Psi\big( w_{1\sim i-1},y_{1\sim i-2},y_{i-1}=j^{\prime}\big)+T_{j^{\prime},j}+E_{i,j}\bigg)\Bigg] \\\\&=log\Bigg[\sum_{j^{\prime}}exp(T_{j\prime,j}+E_{i,j})\sum_{y\in P^{i-2}} exp\bigg( \Psi\big( w_{1\sim i-1},y_{1\sim i-2},y_{i-1}=j^{\prime}\big)\bigg)\Bigg]\\\\&=log\Bigg[\sum_{j^{\prime}}exp(T_{j^{\prime},j}+E_{i,j})exp(\alpha_{i-1,j^{\prime}})\Bigg]\\\\&=log\Bigg[\sum_{j^{\prime}}exp(\alpha_{i-1,j^{\prime}}+T_{j^{\prime},j}+E_{i,j})\Bigg]\end{aligned}$$



  

  ## 代码参考

  * https://github.com/visionshao/LSTM-CRF
  * https://github.com/Magiccircuit/FudanNLP_Begginer
  * https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#bi-lstm-conditional-random-field-discussion (pytorch 官方代码，不支持 batch操作)



