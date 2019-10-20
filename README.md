# 电影短评情感分析

## 数据集说明

data目录里面的数据，只有2万行，完整的数据可关注[斗码小院]公众号，回复"情感分析数据集"获取。

## 代码说明

本项目为电影短评情感分析项目，代码文件说明：

1. [code/preprocessing.py](code/preprocessing.py): 预处理文件
2. [code/stacking.py](code/stacking.py): stacking模型融合项目
3. [code/sentiment_analysis.py](code/sentiment_analysis.py): 基于传统的机器学习算法的情感分析模型。
4. [code/dl_sa.py](code/dl_sa.py): 基于深度学习的情感分析模型。

## 模型结果

![电影短评情感分析：各大模型江湖再见]http://www.csuldw.com/2019/10/19/2019-10-19-comment-analysis/

|模型名称|Accuracy|Precision|Recall|F1-Score|AUC|
|-|-|-|-|-|-|
| word-level-tfidf-LR|0.872|0.87|0.87|0.87|0.9438|
| word-level-tfidf-MNB|0.862|0.86|0.86|0.86|0.9394|
| word-level-tfidf-RF|0.8219|0.82|0.82|0.82|0.8930|
| word-level-tfidf-GBDT|0.723|0.72|0.72|0.71|0.8183|
| word-ngram-tfidf-LR|0.8724|0.87|0.87|0.87|0.9439|
| word-ngram-tfidf-MNB|0.8642|0.86|0.86|0.86|0.9399|
| word-ngram-tfidf-RF|0.8212|0.82|0.82|0.82|0.8925|
| word-ngram-tfidf-GBDT|0.7630|0.77|0.76|0.76|0.8588|
| **char-ngram-tfidf-LR**|**0.8866**|**0.89**|**0.89**|**0.89**|**0.9552**|
| char-ngram-tfidf-MNB|0.8657|0.87|0.87|0.87|0.9410|
| char-ngram-tfidf-RF|0.8276|0.83|0.83|0.83|0.9009|
| char-ngram-tfidf-GBDT|0.7686|0.78|0.77|0.77|0.8613|


## Contributor

<!-- [MIT](LICENSE) &copy;  -->
1. [Diwei Liu](http://www.csuldw.com)
