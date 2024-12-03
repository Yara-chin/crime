#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os

path = r'C:\Users\86173\Downloads\sf-crime\train.csv'  
os.chdir(path)

df= pd.read_csv('train6.csv', encoding='gbk')
pd.set_option('display.max_columns', None)
df = df.dropna(axis=0, how='any')
# 获取列数
num_cols = df.shape[1]
# 去除最后一列（索引为num_cols - 1）
df = df.drop(df.columns[num_cols - 1], axis=1)
print(df.head())
print(df.shape)


# In[2]:


# 2、数据预处理，对category进行编码
from sklearn import preprocessing
label = preprocessing.LabelEncoder()
crime = label.fit_transform(df.Category) #进行编号


# In[3]:


# 先将Dates列转换为日期时间类型
df['Dates'] = pd.to_datetime(df['Dates'])
# 3、对Dates、DayOfWeek、PdDistrict三个特征进行二值化处理，因为3个在训练集和测试集都出现
days = pd.get_dummies(df.DayOfWeek)
district = pd.get_dummies(df.PdDistrict)
hour = pd.get_dummies(df.Dates.dt.hour)



# In[4]:


df_data = pd.concat([days, district, hour], axis=1)   # 将days district hour连成一张表 ，当axis = 1的时候，concat就是行对齐，然后将不同列名称的两张表合并
df_data['crime'] = crime  # 在DataFrame数据结构表的最后加一列，相当于标签
# 实际上，只使用了三个特征，和犯罪类型作为标签 即只使用了原始数据集中的4列数据   
# 但是df_data这张表  其实是将3个特征展开成了几十个特征,对应一个标签


# In[5]:


print(df_data.columns)


# In[6]:


import warnings
warnings.filterwarnings("ignore", message="`np.int` is a deprecated alias for the builtin `int`.")


# In[7]:


import numpy as np
# 4、将样本分割成训练集、测试集和验证集

# df_data包含days、district、hour特征列以及犯罪类型列
import pandas as pd


y = df_data['crime']
threshold = 20
# 先通过loc基于原始df_data进行筛选和赋值操作
# 获取y的唯一值以及对应的计数
unique_y, counts_y = np.unique(y, return_counts=True)
# 构建布尔索引，明确基于唯一值和计数来判断是否小于阈值
bool_index = np.array([counts_y[np.where(unique_y == val)][0] < threshold for val in y])
# 使用构建好的布尔索引通过loc进行筛选和赋值操作
df_data.loc[bool_index, 'crime'] = '其他'

# 1. 先获取之前LabelEncoder编码后的映射关系
#之前的编码是基于df_data中的'Category'列进行的
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(df.Category)
classes_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# 2. 找出一个合适的未使用过的编码值来分配给'其他'
new_encoding_value = max(classes_mapping.values()) + 1 if classes_mapping else 0

# 3. 遍历crime列，根据值是否为'其他'进行相应转换
def convert_crime_value(val):
    if val == '其他':
        return new_encoding_value
    return val

df_data['crime'] = df_data['crime'].apply(convert_crime_value)

# 4. （可选）将crime列转换为整数类型（确保最终数据类型统一）
df_data['crime'] = df_data['crime'].astype(int)

# 重新提取X和y
X = df_data[
    ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday',
             'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN',
             'TARAVAL', 'TENDERLOIN', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
]
y = df_data['crime']


# In[8]:


print(y.value_counts())


# In[9]:


import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

# 先找出样本数量小于2的类别
min_count_classes = y.value_counts()[y.value_counts() < 6].index

# 删除这些类别对应的样本
y = y[~y.isin(min_count_classes)]
# 根据更新后的y，同步删除X中对应的样本（假设X是和y对应的特征数据，同样是DataFrame类型）
X = X.loc[y.index]

# 设置划分比例
train_size = 0.6
valid_size = 0.2
test_size = 0.2

# 使用分层抽样进行数据集划分
sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

# 先划分出训练集和剩余集（包含验证集和测试集）
for train_index, remaining_index in sss.split(X, y):
    X_train, X_remaining = X.iloc[train_index], X.iloc[remaining_index]
    y_train, y_remaining = y.iloc[train_index], y.iloc[remaining_index]

# 对提取出的剩余集X_remaining进行数据类型转换（统一为np.int32类型）
X_remaining = X_remaining.astype(np.int32)

# 再在剩余集中划分验证集和测试集，此时需要重新计算验证集的相对比例
valid_relative_size = valid_size / (valid_size + test_size)
sss_remaining = StratifiedShuffleSplit(n_splits=1, test_size=1 - valid_relative_size, random_state=42)

# 在划分之前检查每个类别的样本数量，确保每个类别至少有两个样本
def safe_split(X, y, sss):
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # 检查并保留样本数量不足的类别
        for label in y.unique():
            if (y_test == label).sum() < 2:
                X_train = pd.concat([X_train, X_test[(y_test == label)]])
                y_train = pd.concat([y_train, y_test[(y_test == label)]])
                X_test = X_test[(y_test != label)]
                y_test = y_test[(y_test != label)]
        yield X_train, X_test

# 使用安全划分函数
for valid_index, test_index in sss_remaining.split(X_remaining, y_remaining):
    X_valid, X_test = X_remaining.iloc[valid_index], X_remaining.iloc[test_index]
    y_valid, y_test = y_remaining.iloc[valid_index], y_remaining.iloc[test_index]

# 查看划分后各个数据集的大小
print("训练集大小:", X_train.shape[0])
print("验证集大小:", X_valid.shape[0])
print("测试集大小:", X_test.shape[0])


# In[10]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import numpy as np


# In[15]:


# 将DataFrame转换为交易列表，并将所有索引转换为字符串类型
transactions = df_data.drop('crime', axis=1).apply(lambda x: [str(index) for index in x[x > 0].index], axis=1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
rule_counts = []  # 新增列表，用于记录每次实验得到的关联规则数量

for train_index, test_index in kf.split(transactions):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 训练集
    df_train = df.iloc[train_index]
    # 测试集
    df_test = df.iloc[test_index]
    
    # Apriori算法
    frequent_itemsets = apriori(df_train, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
    # 记录本次实验得到的关联规则数量
    rule_counts.append(len(rules))   
    # 计算性能指标
    results.append({
        'support': rules['support'].mean(),
        'confidence': rules['confidence'].mean(),
        'lift': rules['lift'].mean()
    })
# 计算平均值和标准差
avg_results = {key: np.mean([result[key] for result in results]) for key in results[0].keys()}
std_results = {key: np.std([result[key] for result in results]) for key in results[0].keys()}

# 绘制支持度、置信度和提升度的分布图
plt.figure(figsize=(12, 6))
sns.histplot(rules['support'], kde=True, label='Support', color='blue')
sns.histplot(rules['confidence'], kde=True, label='Confidence', color='green')
sns.histplot(rules['lift'], kde=True, label='Lift', color='red')
plt.title('Distribution of Support, Confidence, and Lift')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
# 汇总表
summary_df = pd.DataFrame({'Metric': ['Support', 'Confidence', 'Lift'],
                           'Average': [avg_results['support'], avg_results['confidence'], avg_results['lift']],
                           'Std Dev': [std_results['support'], std_results['confidence'], std_results['lift']]})

print(summary_df)

# 绘制汇总图表
plt.figure(figsize=(8, 6))
sns.barplot(x='Metric', y='Average', data=summary_df, yerr=summary_df['Std Dev'], palette='pastel')
plt.title('Average Performance Metrics with Standard Deviation')
plt.ylabel('Average Value')
plt.show()
plt.savefig('performance_metrics.png', dpi=300)
# 可视化展示关联规则数量的直方图
plt.figure(figsize=(8, 6))
sns.histplot(rule_counts, kde=True, bins=10, color='purple')  # 调整bins参数可改变直方图柱子数量，可根据实际情况调整
plt.title('Distribution of Association Rule Counts')
plt.xlabel('Number of Association Rules')
plt.ylabel('Frequency')
plt.show()


# In[12]:


#  rules 是通过 association_rules 函数得到的 DataFrame
num_of_rules = len(rules)
print(f"Number of association rules found: {num_of_rules}")


# In[13]:


# Apriori算法
frequent_itemsets = apriori(df_train, min_support = 0.01, use_colnames = True)
rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.1)

# 展示前10个关联规则
print("前10个关联规则：")
for i in range(10):
    antecedents = ', '.join(list(rules.iloc[i]['antecedents']))
    consequents = ', '.join(list(rules.iloc[i]['consequents']))
    print(f"规则{i + 1}: {antecedents} -> {consequents}")


# In[14]:


# 绘制散点图，展示支持度和置信度的关系
plt.figure(figsize=(10, 8))
sns.scatterplot(x='support', y='confidence', data=rules, hue='lift', palette='viridis', s=100, alpha=0.7)
plt.title('Scatter Plot of Support vs Confidence')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.show()


# In[ ]:




