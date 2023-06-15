# 中医处方生成

1. 融合ChatGPT嵌入
2. 为症状图、草药图增加注意力层

## 依赖库

```bash
conda install pyg -c pyg
torch == 1.5.0+cpu conda install pytorch==1.5.0  cpuonly -c pytorch
torch-geometric == 1.4.3
numpy == 1.18.1
pandas == 1.0.1
sklearn == 0.22.1
```

MUGCN 获得与症状相关的草药 h 的嵌入，然后利用草药 h 的嵌入通过邻近聚集获得当前症状的嵌入。

S-S 图(H-H 图)是根据处方中症状(草药)的共现频率构建的。

SS 图中有158个节点，1273条边
HH 图中有282个节点，4518条边
SH 图中有1195个节点，39935条边

## 人工处方 VS ChatGPT

将症状输入到ChatGPT中进行生成，生成的处方与人工处方进行对比。对比的指标包括草药数量、草药长度、草药的性味归经，还有ChatGPT生成的准确率。

### 草药数量

![人工处方草药数量](output/prescription_test.png)
人工处方的草药数量更多，更加的丰富。

### 草药长度

![人工处方 VS ChatGPT 草药长度对比](output/prescription_test_len.png)
人工处方的包含的草药长度更长。

### 准确率

![人工处方 VS ChatGPT 准确率对比](output/prescription_test_p.png)
ChatGPT生成的处方准确率很低，效果差。

### 草药的性味归经

![人工处方 VS ChatGPT 草药的性味归经对比](https://cdn.jsdelivr.net/gh/Step2312/picgo/202304271057477.png)
人工处方部分草药经过炮制，性味归经的变化，如果不区分炮制和未炮制的草药，ChatGPT生成的准确率会高一点。

## 参考文献

1. Multi-layer information fusion based on graph convolutional network for knowledge-driven herb recommendation
