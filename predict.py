import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#读取数据
x = pd.read_csv("train.csv")
x = x[:100]
print(x.columns)
#数据预处理
#包含缺失值的处理，对一些离散特征进行独热编码，特征选择（降维）？，数据分割
#x.pop('id') #去掉id列,id没什么用，，，，训练集没有id，测试集才有，，，
#特征映射，将朝向转化成编码
# trans_mapping = {"东":1,"南":2,"西":3,"北":4,     
#                  "东北":5,"东南":6,"西北":7,"西南":8} #然后要使用独热编码转换
# x['房屋朝向'] = x['房屋朝向'].map(trans_mapping)

# lb = LabelEncoder()
# x['房屋朝向'] = lb .fit_transform(x['房屋朝向'])
oneHot = pd.get_dummies(x['房屋朝向'])#直接进行了独热编码，比较简洁
x.pop("房屋朝向")
x.join(oneHot)
# ohe = OneHotEncoder(categorical_features=)

#先要替换中文编码
imp = SimpleImputer(missing_values = 'NaN',
              strategy ='mean') #sklearn提供mean，medium,most_frequent等策略

#训练   

#输出

