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


import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 将时间特征中的日期时间字符串转换为日期时间类型
df['Dates'] = pd.to_datetime(df['Dates'])

# 提取小时、星期几、是否是节假日特征
df['Hour'] = df['Dates'].dt.hour
df['DayOfWeek'] = df['Dates'].dt.day_name()
df['IsHoliday'] = (df['DayOfWeek'].isin(['Saturday', 'Sunday'])).astype(int)
# 选择自变量和因变量列
X = df[['X', 'Y', 'PdDistrict', 'Hour', 'DayOfWeek', 'IsHoliday']]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Category'])

# 对分类变量（警区、星期几）进行独热编码
categorical_cols = ['PdDistrict', 'DayOfWeek']
preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    remainder='passthrough'
)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义逻辑回归模型管道
model = make_pipeline(preprocessor, StandardScaler(with_mean=False), LogisticRegression(multi_class='multinomial', solver='lbfgs'))

# 设置交叉验证方式
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 用于存储每次实验的验证分数
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# 进行多次实验
for _ in range(3):
    accuracy_scores.extend(cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy'))
    precision_scores.extend(cross_val_score(model, X_train, y_train, cv=kf, scoring='precision_macro'))
    recall_scores.extend(cross_val_score(model, X_train, y_train, cv=kf, scoring='recall_macro'))
    f1_scores.extend(cross_val_score(model, X_train, y_train, cv=kf, scoring='f1_macro'))

# 计算各性能指标的平均值和标准差
metrics = {
    'Accuracy': accuracy_scores,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1 Score': f1_scores
}

means = {metric: np.mean(scores) for metric, scores in metrics.items()}
stds = {metric: np.std(scores) for metric, scores in metrics.items()}
# 辅助函数，用于在柱状图上添加数值标签
def autolabel(rects, ax):
    """"Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
# 可视化展示各关键性能指标
metrics_list = list(metrics.keys())
index = np.arange(len(metrics_list))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
for i, (metric, scores) in enumerate(metrics.items()):
    rects = ax.bar(index + i * bar_width, means[metric], bar_width, yerr=stds[metric], label=metric, capsize=5)

    # 添加数值标签
    autolabel(rects, ax)


ax.set_xlabel('Performance Metrics')
ax.set_ylabel('Scores')
ax.set_title('Performance Metrics with Mean and Standard Deviation')
ax.set_xticks(index)
ax.set_xticklabels(metrics)
ax.legend()

plt.tight_layout()
plt.show()


# 在测试集上进行预测
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 绘制混淆矩阵
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(30, 24))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig("Confusion Matrix_modle.png", dpi=300)
plt.show()

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"测试集上准确率: {accuracy}")
print(f"测试集上精确率（宏平均）: {precision}")
print(f"测试集上召回率（宏平均）: {recall}")
print(f"测试集上F1值（宏平均）: {f1}")

# 绘制学习曲线
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label='Training Accuracy')
plt.plot(train_sizes, test_scores_mean, label='Validation Accuracy')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.show()




# In[4]:


#展示回归分析的结果
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
from sklearn.linear_model import LinearRegression  # 引入线性回归模型用于回归分析
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score  # 引入回归分析常用的评估指标
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 假设这里将原本的分类任务转变为回归任务，以犯罪发生地点的X坐标值作为因变量
y = df['X']

# 选择自变量列（去除之前用于分类任务中作为因变量的'Category'列等，根据实际合理调整自变量）
X = df[['Y', 'PdDistrict', 'Hour', 'DayOfWeek', 'IsHoliday']]

# 对分类变量（警区、星期几）进行独热编码
categorical_cols = ['PdDistrict', 'DayOfWeek']
preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    remainder='passthrough'
)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义线性回归模型管道
model = make_pipeline(preprocessor, StandardScaler(with_mean=False), LinearRegression())

# 设置交叉验证方式
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 用于存储每次实验的验证分数（这里以均方误差为例，可根据需要添加其他指标如R2等）
mse_scores = []
r2_scores = []

# 进行多次实验（这里进行3次交叉验证，收集均方误差和R2分数）
for _ in range(3):
    mse_cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    mse_scores.extend([-score for score in mse_cv_scores])  # 注意要将neg_mean_squared_error取反得到实际的均方误差
    r2_cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    r2_scores.extend(r2_cv_scores)

# 计算各性能指标的平均值和标准差
metrics = {
    'Mean Squared Error': mse_scores,
    'R2 Score': r2_scores
}
means = {metric: np.mean(scores) for metric, scores in metrics.items()}
stds = {metric: np.std(scores) for metric, scores in metrics.items()}

# 可视化展示各关键性能指标（绘制柱状图及误差线）
metrics_list = list(metrics.keys())
index = np.arange(len(metrics_list))
bar_width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
# 辅助函数，用于在柱状图上添加数值标签
def autolabel(rects, ax):
    """"Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
for i, (metric, scores) in enumerate(metrics.items()):
    rects = ax.bar(index + i * bar_width, means[metric], bar_width, yerr=stds[metric], label=metric, capsize=5)
    # 添加数值标签
    autolabel(rects, ax)

ax.set_xlabel('Performance Metrics')
ax.set_ylabel('Scores')
ax.set_title('Regression Performance Metrics with Mean and Standard Deviation')
ax.set_xticks(index)
ax.set_xticklabels(metrics)
ax.legend()

plt.tight_layout()
plt.show()

# 在测试集上进行预测
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 计算测试集上的均方误差和R2分数
mse_test = mean_squared_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)
print(f"测试集上均方误差: {mse_test}")
print(f"测试集上R2分数: {r2_test}")

# 绘制预测值与真实值的散点图，查看拟合情况
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values in Regression')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 绘制对角线作为参考线
plt.show()

# 绘制学习曲线（展示训练集和验证集上的均方误差随训练样本数变化情况）
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
)

train_scores_mean = -np.mean(train_scores, axis=1)  # 取反得到实际均方误差
test_scores_mean = -np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label='Training MSE')
plt.plot(train_sizes, test_scores_mean, label='Validation MSE')
plt.xlabel('Training Set Size')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curve for Regression')
plt.legend()
plt.show()



# In[ ]:




