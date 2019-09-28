
#导入所需模块

# -*- coding: utf-8  -*-
import warnings
warnings.filterwarnings("ignore")
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10, 5)
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[ ]:数据准备，读取文件，数据预处理阶段包括：整合数据生成样本空间、生成词袋、统计词频生成模
import os
#获取目标文件夹的路径
filedir ='C:\\Users\\ccy\\Desktop\\ChnSentiCorp\\ChnSentiCorp\\neg'
#获取当前文件夹中的文件名称列表  
filenames=os.listdir(filedir)
#打开当前目录下的result.txt文件，如果没有则创建
f=open('C:\\Users\\ccy\\Desktop\\ChnSentiCorp\\ChnSentiCorp\\neg.txt','w')
#先遍历文件名
for filename in filenames:
    filepath = filedir+'/'+filename
    #遍历单个文件，读取行数
    for line in open(filepath,encoding='UTF-8'):
        f.writelines(line)
        f.write('\n')
#关闭文件
f.close()


# In[]:#准备停用词表
stopwords = []
for line in open("E:/Jupyter/sentiment/stopwords.txt",encoding="UTF-8"):
    stopwords.append(line.strip())
#生成词袋    
def read_file(fi,sentiment, stopwords, words, sentences):
    for line in open(fi,encoding="UTF-8"):
        try:
            segs = jieba.lcut(line.strip())#结巴分词
            segs = [word for word in segs if word not in stopwords and len(word) > 1]#去掉停用词
            words.extend(segs)
            sentences.append((segs, sentiment)) # tuple，评价+标签
        except:
            print(line)
            continue
words = []
sentences = []
sentiment = 1
read_file('E:/Jupyter/sentiment/pos.txt', sentiment, stopwords, words, sentences)#正向评价及标签
sentiment = 0
read_file('E:/Jupyter/sentiment/neg.txt', sentiment, stopwords, words, sentences)#负向评价及标签


# In[ ]:机器学习方法

# In[]:整合正负向语料
x, y = zip(*sentences) #zip相当与压缩 ，zip（*）相当于解压
x = [" ".join(sentences) for sentences in x]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.1)  #划分训练集和测试集
# 特征提取
vec = CountVectorizer(ngram_range=(1, 2), max_features= 1000) #ngram_range=(1, 2)选用1,2个词进行前后的组合，构成新的标签值，#CountVectorizer只考虑每种词汇在该训练文本中出现的频。将文本向量转换成稀疏表示数值向量（字符频率向量）。max_feature 选取频率高的单词
vec.fit(x_train)
classifier = MultinomialNB() #基于多项式的朴素贝叶
classifier.fit(vec.transform(x_train), y_train) #tranform()的作用是通过找中心和缩放等实现标准化
# 测试得分
classifier.score(vec.transform(x_test), y_test)


# In[]:深度学习方法
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
import pandas as pd

tokenizer = Tokenizer(nb_words=2500, split=' ') # 用于向量化文本,或将文本转换为序列，保留2500个词
tokenizer.fit_on_texts(x)#生成词典
X = tokenizer.texts_to_sequences(x)#将每条文本转变成一个向量
X = pad_sequences(X)#将序列转化为经过填充以后的一个长度相同的新序列

# 设定embedding维度等超参数
embed_dim = 16
lstm_out = 100
batch_size = 32
# 构建LSTM网络
model = Sequential()
model.add(Embedding(2500, embed_dim, input_length=X.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
Y = pd.get_dummies(pd.DataFrame({'label': [str(target) for target in y]})).values
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1, random_state=7)
# 拟合与训练模型
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=10)
# 验证
score, acc = model.evaluate(X_valid, Y_valid, verbose=2, batch_size=batch_size)
print('Logloss损失: %.2f' %(score))
print('验证集的准确率 :%.2f'%(acc))


# In[ ]细节注意：文本数据如何做预处理，如何清洗数据，对不均衡的类别如何处理，文本数据特征工程方式，Word2vec/word embedding的理解，CNN/LSTM技术细节模型评估，与过拟合解决方法

