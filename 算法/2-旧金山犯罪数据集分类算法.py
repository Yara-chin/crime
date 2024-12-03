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
# 2、数据预处理，对category进行编码
from sklearn import preprocessing
label = preprocessing.LabelEncoder()
crime = label.fit_transform(df.Category) #进行编号
# 先将Dates列转换为日期时间类型
df['Dates'] = pd.to_datetime(df['Dates'])
# 3、对Dates、DayOfWeek、PdDistrict三个特征进行二值化处理，因为3个在训练集和测试集都出现
days = pd.get_dummies(df.DayOfWeek)
district = pd.get_dummies(df.PdDistrict)
hour = pd.get_dummies(df.Dates.dt.hour)
df_data = pd.concat([days, district, hour], axis=1)   # 将days district hour连成一张表 ，当axis = 1的时候，concat就是行对齐，然后将不同列名称的两张表合并
df_data['crime'] = crime  # 在DataFrame数据结构表的最后加一列，相当于标签
# 实际上，只使用了三个特征，和犯罪类型作为标签 即只使用了原始数据集中的4列数据   
# 但是df_data这张表  其实是将3个特征展开成了几十个特征,对应一个标签
print(df_data.columns)
import warnings
warnings.filterwarnings("ignore", message="`np.int` is a deprecated alias for the builtin `int`.")
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

# 1. 先获取之前LabelEncoder编码后的映射关系（如果之前编码结果还能获取到的话）
# 假设之前的编码是基于df_data中的'Category'列进行的
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(df.Category)
classes_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# 2. 找出一个合适的未使用过的编码值来分配给'其他'（这里简单地取编码值的最大值加1，你可根据实际情况调整）
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
print(y.value_counts())
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


# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree


# In[3]:


#决策树

# 假设之前的数据预处理已经得到了X_train, X_valid, X_test, y_train, y_valid, y_test

# 用于存储每次实验的性能指标结果（训练集）
train_accuracy_results = []
train_f1_results = []
train_recall_results = []
train_precision_results = []

# 用于存储每次实验的性能指标结果（测试集）
test_accuracy_results = []
test_f1_results = []
test_recall_results = []
test_precision_results = []

# 设置交叉验证折数
n_splits = 5
# 重复实验次数
num_experiments = 3

# 参数搜索范围
min_samples_leaf_options = [25, 50, 100]
max_depth_options = [3, 5, 7]

for experiment in range(num_experiments):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_train_accuracy = []
    fold_train_f1 = []
    fold_train_recall = []
    fold_train_precision = []
    fold_test_accuracy = []
    fold_test_f1 = []
    fold_test_recall = []
    fold_test_precision = []

    for min_samples_leaf in min_samples_leaf_options:
        for max_depth in max_depth_options:
            for train_index, test_index in skf.split(X_train, y_train):
                X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
                y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

                # 构建决策树分类模型
                clf = DecisionTreeClassifier(random_state = 42, min_samples_leaf = min_samples_leaf, max_depth = max_depth, class_weight='balanced')
                clf.fit(X_train_fold, y_train_fold)

                # 预测训练集
                y_train_pred = clf.predict(X_train_fold)
                # 预测测试集
                y_test_pred = clf.predict(X_test_fold)

                # 计算训练集性能指标
                train_accuracy = accuracy_score(y_train_fold, y_train_pred)
                train_f1 = f1_score(y_train_fold, y_train_pred, average='weighted')
                train_recall = recall_score(y_train_fold, y_train_pred, average='weighted')
                train_precision = precision_score(y_train_fold, y_train_pred, average='weighted')

                # 计算测试集性能指标
                test_accuracy = accuracy_score(y_test_fold, y_test_pred)
                test_f1 = f1_score(y_test_fold, y_test_pred, average='weighted')
                test_recall = recall_score(y_test_fold, y_test_pred, average='weighted')
                test_precision = precision_score(y_test_fold, y_test_pred, average='weighted')

                fold_train_accuracy.append(train_accuracy)
                fold_train_f1.append(train_f1)
                fold_train_recall.append(train_recall)
                fold_train_precision.append(train_precision)
                fold_test_accuracy.append(test_accuracy)
                fold_test_f1.append(test_f1)
                fold_test_recall.append(test_recall)
                fold_test_precision.append(test_precision)

    train_accuracy_results.append(fold_train_accuracy)
    train_f1_results.append(fold_train_f1)
    train_recall_results.append(fold_train_recall)
    train_precision_results.append(fold_train_precision)
    test_accuracy_results.append(fold_test_accuracy)
    test_f1_results.append(fold_test_f1)
    test_recall_results.append(fold_test_recall)
    test_precision_results.append(fold_test_precision)

# 计算训练集性能指标的平均值和标准差
train_accuracy_mean = np.mean([np.mean(i) for i in train_accuracy_results])
train_accuracy_std = np.std([np.mean(i) for i in train_accuracy_results])
train_f1_mean = np.mean([np.mean(i) for i in train_f1_results])
train_f1_std = np.std([np.mean(i) for i in train_f1_results])
train_recall_mean = np.mean([np.mean(i) for i in train_recall_results])
train_recall_std = np.std([np.mean(i) for i in train_recall_results])
train_precision_mean = np.mean([np.mean(i) for i in train_precision_results])
train_precision_std = np.std([np.mean(i) for i in train_precision_results])

# 计算测试集性能指标的平均值和标准差
test_accuracy_mean = np.mean([np.mean(i) for i in test_accuracy_results])
test_accuracy_std = np.std([np.mean(i) for i in test_accuracy_results])
test_f1_mean = np.mean([np.mean(i) for i in test_f1_results])
test_f1_std = np.std([np.mean(i) for i in test_f1_results])
test_recall_mean = np.mean([np.mean(i) for i in test_recall_results])
test_recall_std = np.std([np.mean(i) for i in test_recall_results])
test_precision_mean = np.mean([np.mean(i) for i in test_precision_results])
test_precision_std = np.std([np.mean(i) for i in test_precision_results])

# 使用最佳参数构建最终决策树
best_min_samples_leaf = min_samples_leaf_options[0]
best_max_depth = max_depth_options[0]
final_clf = DecisionTreeClassifier(random_state=42, min_samples_leaf=best_min_samples_leaf, max_depth=best_max_depth)
final_clf.fit(X_train, y_train)

# 可视化决策树结构
plt.figure(figsize=(15, 10))
from sklearn.tree import plot_tree
plot_tree(final_clf, filled=True, feature_names=X_train.columns, class_names=np.unique(y_train).astype(str))
plt.title("Decision Tree Structure")
plt.savefig("decision_tree_structure.png", dpi=300)
plt.show()

# 应用最终模型对测试集进行预测，获取最终测试集性能指标
final_y_pred_test = final_clf.predict(X_test)
final_test_accuracy = accuracy_score(y_test, final_y_pred_test)
final_test_f1 = f1_score(y_test, final_y_pred_test, average='weighted')
final_test_recall = recall_score(y_test, final_y_pred_test, average='weighted')
final_test_precision = precision_score(y_test, final_y_pred_test, average='weighted')

# 制作汇总表
summary_table = pd.DataFrame({
    'Metric': ['Accuracy', 'F - measure', 'Recall', 'Precision'],
    'Train Mean': [train_accuracy_mean, train_f1_mean, train_recall_mean, train_precision_mean],
    'Train Std': [train_accuracy_std, train_f1_std, train_recall_std, train_precision_std],
    'Test Mean': [test_accuracy_mean, test_f1_mean, test_recall_mean, test_precision_mean],
    'Test Final': [final_test_accuracy, final_test_f1, final_test_recall, final_test_precision]
})
print(summary_table)

# 可视化展示汇总表（柱状图 + 误差线）
plt.figure(figsize=(10, 8))
metrics = ['Accuracy', 'F - measure', 'Recall', 'Precision']
bar_width = 0.35
bar_positions_train_mean = np.arange(len(metrics))
bar_positions_test_mean = [pos + bar_width for pos in bar_positions_train_mean]

plt.bar(bar_positions_train_mean, summary_table['Train Mean'], width=bar_width, label='Train Mean', yerr=summary_table['Train Std'])
plt.bar(bar_positions_test_mean, summary_table['Test Mean'], width=bar_width, label='Test Mean')
plt.scatter(bar_positions_test_mean, summary_table['Test Final'], color='red', label='Test Final')

plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Performance Metrics Comparison')
plt.xticks([pos + bar_width / 2 for pos in bar_positions_train_mean], metrics)
plt.legend()
plt.savefig("summary_performance_metrics.png", dpi=300)
plt.show()


# In[4]:


#随机森林
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt


# 用于存储每次实验的性能指标结果（训练集）
train_accuracy_results = []
train_f1_results = []
train_recall_results = []
train_precision_results = []

# 用于存储每次实验的性能指标结果（测试集）
test_accuracy_results = []
test_f1_results = []
test_recall_results = []
test_precision_results = []

# 设置交叉验证折数
n_splits = 5
# 重复实验次数
num_experiments = 3

# 参数搜索范围
min_samples_leaf_options = [1, 5, 10]
max_depth_options = [3, 5, 7]

for experiment in range(num_experiments):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_train_accuracy = []
    fold_train_f1 = []
    fold_train_recall = []
    fold_train_precision = []
    fold_test_accuracy = []
    fold_test_f1 = []
    fold_test_recall = []
    fold_test_precision = []

    for min_samples_leaf in min_samples_leaf_options:
        for max_depth in max_depth_options:
            for train_index, test_index in skf.split(X_train, y_train):
                X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
                y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

                # 构建随机森林分类模型
                clf = RandomForestClassifier(random_state=42, min_samples_leaf=min_samples_leaf, max_depth=max_depth)
                clf.fit(X_train_fold, y_train_fold)

                # 预测训练集
                y_train_pred = clf.predict(X_train_fold)
                # 预测测试集
                y_test_pred = clf.predict(X_test_fold)

                # 计算训练集性能指标
                train_accuracy = accuracy_score(y_train_fold, y_train_pred)
                train_f1 = f1_score(y_train_fold, y_train_pred, average='weighted')
                train_recall = recall_score(y_train_fold, y_train_pred, average='weighted')
                train_precision = precision_score(y_train_fold, y_train_pred, average='weighted')

                # 计算测试集性能指标
                test_accuracy = accuracy_score(y_test_fold, y_test_pred)
                test_f1 = f1_score(y_test_fold, y_test_pred, average='weighted')
                test_recall = recall_score(y_test_fold, y_test_pred, average='weighted')
                test_precision = precision_score(y_test_fold, y_test_pred, average='weighted')

                fold_train_accuracy.append(train_accuracy)
                fold_train_f1.append(train_f1)
                fold_train_recall.append(train_recall)
                fold_train_precision.append(train_precision)
                fold_test_accuracy.append(test_accuracy)
                fold_test_f1.append(test_f1)
                fold_test_recall.append(test_recall)
                fold_test_precision.append(test_precision)

    train_accuracy_results.append(fold_train_accuracy)
    train_f1_results.append(fold_train_f1)
    train_recall_results.append(fold_train_recall)
    train_precision_results.append(fold_train_precision)
    test_accuracy_results.append(fold_test_accuracy)
    test_f1_results.append(fold_test_f1)
    test_recall_results.append(fold_test_recall)
    test_precision_results.append(fold_test_precision)

# 计算训练集性能指标的平均值和标准差
train_accuracy_mean = np.mean([np.mean(i) for i in train_accuracy_results])
train_accuracy_std = np.std([np.mean(i) for i in train_accuracy_results])
train_f1_mean = np.mean([np.mean(i) for i in train_f1_results])
train_f1_std = np.std([np.mean(i) for i in train_f1_results])
train_recall_mean = np.mean([np.mean(i) for i in train_recall_results])
train_recall_std = np.std([np.mean(i) for i in train_recall_results])
train_precision_mean = np.mean([np.mean(i) for i in train_precision_results])
train_precision_std = np.std([np.mean(i) for i in train_precision_results])

# 计算测试集性能指标的平均值和标准差
test_accuracy_mean = np.mean([np.mean(i) for i in test_accuracy_results])
test_accuracy_std = np.std([np.mean(i) for i in test_accuracy_results])
test_f1_mean = np.mean([np.mean(i) for i in test_f1_results])
test_f1_std = np.std([np.mean(i) for i in test_f1_results])
test_recall_mean = np.mean([np.mean(i) for i in test_recall_results])
test_recall_std = np.std([np.mean(i) for i in test_recall_results])
test_precision_mean = np.mean([np.mean(i) for i in test_precision_results])
test_precision_std = np.std([np.mean(i) for i in test_precision_results])

# 使用最佳参数构建最终随机森林模型
best_min_samples_leaf = min_samples_leaf_options[0]
best_max_depth = max_depth_options[0]
final_clf = RandomForestClassifier(random_state=42, min_samples_leaf=best_min_samples_leaf, max_depth=best_max_depth)
final_clf.fit(X_train, y_train)

# 可视化随机森林模型（展示特征重要性）
importances = final_clf.feature_importances_
feature_names = X_train.columns.astype(str)
plt.figure(figsize=(10, 5))
plt.bar(feature_names, importances)
plt.xticks(rotation=90)
plt.title("Feature Importances in Random Forest")
plt.savefig("feature_importances.png", dpi=300)
plt.show()

# 应用最终模型对测试集进行预测，获取最终测试集性能指标
final_y_pred_test = final_clf.predict(X_test)
final_test_accuracy = accuracy_score(y_test, final_y_pred_test)
final_test_f1 = f1_score(y_test, final_y_pred_test, average='weighted')
final_test_recall = recall_score(y_test, final_y_pred_test, average='weighted')
final_test_precision = precision_score(y_test, final_y_pred_test, average='weighted')

# 制作汇总表
summary_table = pd.DataFrame({
    'Metric': ['Accuracy', 'F - measure', 'Recall', 'Precision'],
    'Train Mean': [train_accuracy_mean, train_f1_mean, train_recall_mean, train_precision_mean],
    'Train Std': [train_accuracy_std, train_f1_std, train_recall_std, train_precision_std],
    'Test Mean': [test_accuracy_mean, test_f1_mean, test_recall_mean, test_precision_mean],
    'Test Final': [final_test_accuracy, final_test_f1, final_test_recall, final_test_precision]
})
print(summary_table)

# 可视化展示汇总表（柱状图 + 误差线）
plt.figure(figsize=(10, 8))
metrics = ['Accuracy', 'F - measure', 'Recall', 'Precision']
bar_width = 0.35
bar_positions_train_mean = np.arange(len(metrics))
bar_positions_test_mean = [pos + bar_width for pos in bar_positions_train_mean]

plt.bar(bar_positions_train_mean, summary_table['Train Mean'], width=bar_width, label='Train Mean', yerr=summary_table['Train Std'])
plt.bar(bar_positions_test_mean, summary_table['Test Mean'], width=bar_width, label='Test Mean')
plt.scatter(bar_positions_test_mean, summary_table['Test Final'], color='red', label='Test Final')

plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Performance Metrics Comparison')
plt.xticks([pos + bar_width / 2 for pos in bar_positions_train_mean], metrics)
plt.legend()
plt.savefig("summary_performance_metrics.png", dpi=300)
plt.show()


# In[5]:


# 生成随机森林学习曲线
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(
    final_clf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# 绘制学习曲线
plt.figure(figsize=(10, 8))
plt.title("Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross - validation score")

plt.legend(loc="best")
plt.savefig("learning_curve.png", dpi=300)
plt.show()


# In[6]:


#朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns


# 用于存储每次实验的性能指标结果（训练集）
train_accuracy_results = []
train_f1_results = []
train_recall_results = []
train_precision_results = []

# 用于存储每次实验的性能指标结果（测试集）
test_accuracy_results = []
test_f1_results = []
test_recall_results = []
test_precision_results = []

# 设置交叉验证折数
n_splits = 5
# 重复实验次数
num_experiments = 3

# 初始化朴素贝叶斯分类器并使用BaggingClassifier进行集成
nb = GaussianNB()
bagging_nb = BaggingClassifier(base_estimator=nb, n_estimators=10, random_state=42)

for experiment in range(num_experiments):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_train_accuracy = []
    fold_train_f1 = []
    fold_train_recall = []
    fold_train_precision = []
    fold_test_accuracy = []
    fold_test_f1 = []
    fold_test_recall = []
    fold_test_precision = []

    for train_index, test_index in skf.split(X_train, y_train):
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

        # 训练模型
        bagging_nb.fit(X_train_fold, y_train_fold)

        # 预测训练集
        y_train_pred = bagging_nb.predict(X_train_fold)
        # 预测测试集
        y_test_pred = bagging_nb.predict(X_test_fold)

        # 计算训练集性能指标
        train_accuracy = accuracy_score(y_train_fold, y_train_pred)
        train_f1 = f1_score(y_train_fold, y_train_pred, average='weighted')
        train_recall = recall_score(y_train_fold, y_train_pred, average='weighted')
        train_precision = precision_score(y_train_fold, y_train_pred, average='weighted')

        # 计算测试集性能指标
        test_accuracy = accuracy_score(y_test_fold, y_test_pred)
        test_f1 = f1_score(y_test_fold, y_test_pred, average='weighted')
        test_recall = recall_score(y_test_fold, y_test_pred, average='weighted')
        test_precision = precision_score(y_test_fold, y_test_pred, average='weighted')

        fold_train_accuracy.append(train_accuracy)
        fold_train_f1.append(train_f1)
        fold_train_recall.append(train_recall)
        fold_train_precision.append(train_precision)
        fold_test_accuracy.append(test_accuracy)
        fold_test_f1.append(test_f1)
        fold_test_recall.append(test_recall)
        fold_test_precision.append(test_precision)

    train_accuracy_results.append(fold_train_accuracy)
    train_f1_results.append(fold_train_f1)
    train_recall_results.append(fold_train_recall)
    train_precision_results.append(fold_train_precision)
    test_accuracy_results.append(fold_test_accuracy)
    test_f1_results.append(fold_test_f1)
    test_recall_results.append(fold_test_recall)
    test_precision_results.append(fold_test_precision)

# 计算训练集性能指标的平均值和标准差
train_accuracy_mean = np.mean([np.mean(i) for i in train_accuracy_results])
train_accuracy_std = np.std([np.mean(i) for i in train_accuracy_results])
train_f1_mean = np.mean([np.mean(i) for i in train_f1_results])
train_f1_std = np.std([np.mean(i) for i in train_f1_results])
train_recall_mean = np.mean([np.mean(i) for i in train_recall_results])
train_recall_std = np.std([np.mean(i) for i in train_recall_results])
train_precision_mean = np.mean([np.mean(i) for i in train_precision_results])
train_precision_std = np.std([np.mean(i) for i in train_precision_results])

# 计算测试集性能指标的平均值和标准差
test_accuracy_mean = np.mean([np.mean(i) for i in test_accuracy_results])
test_accuracy_std = np.std([np.mean(i) for i in test_accuracy_results])
test_f1_mean = np.mean([np.mean(i) for i in test_f1_results])
test_f1_std = np.std([np.mean(i) for i in test_f1_results])
test_recall_mean = np.mean([np.mean(i) for i in test_recall_results])
test_recall_std = np.std([np.mean(i) for i in test_recall_results])
test_precision_mean = np.mean([np.mean(i) for i in test_precision_results])
test_precision_std = np.std([np.mean(i) for i in test_precision_results])

# 使用全部训练集训练最终模型
bagging_nb.fit(X_train, y_train)

# 预测测试集，获取最终测试集性能指标
final_y_pred_test = bagging_nb.predict(X_test)
final_test_accuracy = accuracy_score(y_test, final_y_pred_test)
final_test_f1 = f1_score(y_test, final_y_pred_test, average='weighted')
final_test_recall = recall_score(y_test, final_y_pred_test, average='weighted')
final_test_precision = precision_score(y_test, final_y_pred_test, average='weighted')

# 制作汇总表
summary_table = pd.DataFrame({
    'Metric': ['Accuracy', 'F - measure', 'Recall', 'Precision'],
    'Train Mean': [train_accuracy_mean, train_f1_mean, train_recall_mean, train_precision_mean],
    'Train Std': [train_accuracy_std, train_f1_std, train_recall_std, train_precision_std],
    'Test Mean': [test_accuracy_mean, test_f1_mean, test_recall_mean, test_precision_mean],
    'Test Final': [final_test_accuracy, final_test_f1, final_test_recall, final_test_precision]
})
print(summary_table)

# 可视化展示汇总表（柱状图 + 误差线）
plt.figure(figsize=(10, 8))
metrics = ['Accuracy', 'F - measure', 'Recall', 'Precision']
bar_width = 0.35
bar_positions_train_mean = np.arange(len(metrics))
bar_positions_test_mean = [pos + bar_width for pos in bar_positions_train_mean]

plt.bar(bar_positions_train_mean, summary_table['Train Mean'], width=bar_width, label='Train Mean', yerr=summary_table['Train Std'])
plt.bar(bar_positions_test_mean, summary_table['Test Mean'], width=bar_width, label='Test Mean')
plt.scatter(bar_positions_test_mean, summary_table['Test Final'], color='red', label='Test Final')

plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Performance Metrics Comparison')
plt.xticks([pos + bar_width / 2 for pos in bar_positions_train_mean], metrics)
plt.legend()
plt.savefig("summary_performance_metrics.png", dpi=300)
plt.show()

# 绘制混淆矩阵
cm = confusion_matrix(y_test, final_y_pred_test)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()


# In[7]:


#朴素贝叶斯学习曲线
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(
    bagging_nb, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 8))
plt.title("Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross - validation score")

plt.legend(loc="best")
plt.savefig("learning_curve.png", dpi=300)
plt.show()


# In[11]:


#计算并绘制宏平均和加权平均的ROC曲线和AUC值
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

# 将标签二值化，以适应roc_curve函数
y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))
y_scores = bagging_nb.predict_proba(X_test)
# 计算宏平均和加权平均的AUC值
roc_auc_macro = roc_auc_score(y_test_binarized, y_scores, multi_class='ovr', average='macro')
roc_auc_weighted = roc_auc_score(y_test_binarized, y_scores, multi_class='ovr', average='weighted')

print(f"Macro-average AUC: {roc_auc_macro}")
print(f"Weighted-average AUC: {roc_auc_weighted}")

# 选择几个代表性的类别来绘制ROC曲线
代表性类别 = [3, 7, 17, ] 
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green']

plt.figure(figsize=(12, 8))

# 绘制选定类别的ROC曲线
for i, color in zip(代表性类别, colors):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'ROC curve of class {i} (area = {auc(fpr, tpr):0.2f})')

# 绘制宏平均ROC曲线
fpr_macro, tpr_macro, _ = roc_curve(y_test_binarized.ravel(), y_scores.ravel())
plt.plot(fpr_macro, tpr_macro, color='purple', lw=2, label='Macro-average ROC curve')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Selected Classes and Macro-average')
plt.legend(loc="lower right")
plt.savefig("multiclass_roc_curve.png", dpi=300)
plt.show()


# In[20]:


# 新增代码部分，绘制随机森林中某一棵树的结构
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
import os
from PIL import Image as PILImage
import io
# 获取随机森林中的第一棵树
tree = final_clf.estimators_[0]

# 将决策树导出为Graphviz格式
dot_data = export_graphviz(tree,
                           out_file=None,
                           feature_names=X_train.columns,
                           class_names=final_clf.classes_.astype(str),
                           filled=True,
                           rounded=True,
                           special_characters=True)

# 使用pydotplus将Graphviz格式数据转换为图像
graph = pydotplus.graph_from_dot_data(dot_data)
# 获取图像的字节流数据
png_bytes = graph.create_png()

# 将字节流数据转换为BytesIO对象，使其能被PIL.Image.open识别并读取
img_bytesio = io.BytesIO(png_bytes)

# 使用PIL.Image.open从BytesIO对象中读取图像数据并转换为numpy数组
img_array = np.array(PILImage.open(img_bytesio))

# 将图像对象转换为matplotlib可处理的格式，并设置分辨率（dpi），这里设置为300，你可以根据需要调整更高的值
fig = plt.figure(figsize=(8, 8), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')
ax.imshow(img_array)

# 保存图片，设置图片质量（可选参数，可根据需要调整），这里使用默认的图片质量
output_path = os.path.join('.', 'tree_structure.png')
fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1, quality=95)




# In[21]:


# 在Jupyter Notebook等环境中展示图像
Image(graph.create_png())


# In[ ]:




