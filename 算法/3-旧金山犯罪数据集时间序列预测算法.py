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
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
import seaborn as sns

# df 是已经加载的 DataFrame
df['Dates'] = pd.to_datetime(df['Dates'])
df.set_index('Dates', inplace=True)

# 计算每天的犯罪次数
daily_crime_counts = df.resample('D').size()

# 检查数据
print(daily_crime_counts.head())

# 数据集划分
train_size = int(len(daily_crime_counts) * 0.8)  # 80%的数据作为训练集
train_data = daily_crime_counts[:train_size]
test_data = daily_crime_counts[train_size:]

# 检查训练数据的平稳性
def test_stationarity(timeseries):
    dftest = adfuller(timeseries, autolag='AIC')
    return dftest[1]  # 返回p - value


p_value = test_stationarity(train_data)
print('ADF Statistic: %f' % adfuller(train_data)[0])
print('p - value: %f' % p_value)

# 差分使序列平稳（训练数据）
train_data_diff = train_data.diff().dropna()
p_value_diff = test_stationarity(train_data_diff)
print('p - value after diff: %f' % p_value_diff)

# 确定ARIMA模型参数（基于训练数据）
import itertools


# 定义p, d, q的范围
p = d = q = range(0, 3)

# 生成所有可能的(p, d, q)组合
pdq = list(itertools.product(p, d, q))

best_aic = float("inf")
best_pdq = None

# 遍历所有可能的(p, d, q)组合
for param in pdq:
    try:
        model = ARIMA(train_data, order=param)
        model_fit = model.fit()
        aic = model_fit.aic
        if aic < best_aic:
            best_aic = aic
            best_pdq = param
    except:
        continue

print(f"Best ARIMA parameters: p={best_pdq[0]}, d={best_pdq[1]}, q={best_pdq[2]}")

# 训练ARIMA模型（基于训练数据）
best_model = ARIMA(train_data, order=best_pdq)
best_model_fit = best_model.fit()

# 时间序列交叉验证（基于训练数据）
def cross_validate_forecast(model, order, X, n_splits=5):
    predictions = []
    actuals = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(X):
        train = X[train_index]
        test = X[test_index]
        if len(train) > 1:
            try:
                model_arima = ARIMA(train, order=order)
                model_fit = model_arima.fit()
                pred = model_fit.forecast(steps=len(test))
                predictions.append(pred)
                actuals.append(test)
            except Exception as e:
                print(f"Error fitting model in fold {i + 1}: {e}")
        else:
            print(f"Insufficient data in fold {i + 1} for model fitting.")
    return predictions, actuals


predictions, actuals = cross_validate_forecast(best_model, best_pdq, train_data)

# 展示模型结构（通过打印参数）
print(f"ARIMA Model Structure: ARIMA({best_pdq[0]}, {best_pdq[1]}, {best_pdq[2]})")

# 计算性能指标
mse_scores = [mean_squared_error(actual, pred) for actual, pred in zip(actuals, predictions)]
mae_scores = [mean_absolute_error(actual, pred) for actual, pred in zip(actuals, predictions)]
rmse_scores = [np.sqrt(mse) for mse in mse_scores]

print(f"MSE scores: {mse_scores}")
print(f"Average MSE: {np.mean(mse_scores)}")
print(f"Standard Deviation of MSE: {np.std(mse_scores)}")
print(f"MAE scores: {mae_scores}")
print(f"Average MAE: {np.mean(mae_scores)}")
print(f"RMSE scores: {rmse_scores}")
print(f"Average RMSE: {np.mean(rmse_scores)}")

# 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Observed')

for i, pred in enumerate(predictions):
    actual = actuals[i]
    if len(pred) > len(actual):
        print(f"Warning: Forecast length ({len(pred)}) is longer than actual length ({len(actual)}) in fold {i + 1}. Trimming prediction to match actual length.")
        pred = pred[:len(actual)]
    elif len(pred) < len(actual):
        print(f"Warning: Forecast length ({len(pred)}) is shorter than actual length ({len(actual)}) in fold {i + 1}. Extending prediction with last value to match actual length.")
        last_value = pred[-1] if len(pred) > 0 else 0
        pred = np.pad(pred, (0, len(actual) - len(pred)), 'constant', constant_values=last_value)

    x_data = np.arange(i * len(actual), i * len(actual) + len(pred))
    plt.plot(x_data, pred, label=f'Forecast {i + 1}')

plt.legend()
plt.title("Observed vs Forecasted Crime Counts (Training Data)")
plt.xlabel("Time")
plt.ylabel("Crime Counts")
plt.show()

# 绘制MSE分数柱状图
mse_df = pd.DataFrame({'MSE': mse_scores})
mse_df['Fold'] = range(1, len(mse_scores) + 1)

plt.figure(figsize=(8, 6))
sns.barplot(x='Fold', y='MSE', data=mse_df, palette='viridis')
plt.title('MSE Scores for Each Cross - Validation Fold (Training Data)')
plt.xlabel('Fold')
plt.ylabel('MSE')
plt.savefig("mse_scores_train.png", dpi=300)
plt.show()

# 在测试集上进行预测
test_predictions = best_model_fit.forecast(steps=len(test_data))
test_mse = mean_squared_error(test_data, test_predictions)
print(f"Test MSE: {test_mse}")

# 可视化测试集预测结果
plt.figure(figsize=(12, 6))
plt.plot(train_data, label='Train')
plt.plot(test_data.index, test_predictions, label='Test Forecast')
plt.plot(test_data, label='Test Observed')
plt.legend()
plt.title("Train vs Test Forecasted Crime Counts")
plt.xlabel("Time")
plt.ylabel("Crime Counts")
plt.show()


# In[3]:


import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# 确定ARIMA模型参数（假设已经确定好最佳参数p, d, q）
p = 2
d = 0
q = 1


def plot_learning_curve():
    train_sizes = np.linspace(0.1, 1.0, 10)
    valid_indices = []
    mse_scores = []
    for index, train_size in enumerate(train_sizes):
        train_size = int(len(daily_crime_counts) * train_size)
        if train_size == 0:
            continue
        train_data = daily_crime_counts[:train_size]
        test_data = daily_crime_counts[train_size:]
        if len(test_data) == 0:
            continue
        model = ARIMA(train_data, order=(p, d, q))
        model_fit = model.fit()
        preds = model_fit.forecast(steps=len(test_data))
        mse = mean_squared_error(test_data, preds)
        mse_scores.append(mse)
        valid_indices.append(index)

    # 根据有效的索引提取对应的train_sizes元素
    final_train_sizes = train_sizes[valid_indices]

    plt.plot(final_train_sizes, mse_scores, marker='o')
    plt.xlabel('Training Data Size')
    plt.ylabel('MSE')
    plt.title('ARIMA Model Learning Curve')
    plt.show()


plot_learning_curve()


# In[4]:


# 可视化模型结构（使用文本框）
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnchoredText

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(daily_crime_counts, label='Observed')

for i, pred in enumerate(predictions):
    actual = actuals[i]
    if len(pred) > len(actual):
        print(f"Warning: Forecast length ({len(pred)}) is longer than actual length ({len(actual)}) in fold {i + 1}. Trimming prediction to match actual length.")
        pred = pred[:len(actual)]
    elif len(pred) < len(actual):
        print(f"Warning: Forecast length ({len(pred)}) is shorter than actual length ({len(actual)}) in fold {i + 1}. Extending prediction with last value to match actual length.")
        last_value = pred[-1] if len(pred) > 0 else 0
        pred = np.pad(pred, (0, len(actual) - len(pred)), 'constant', constant_values=last_value)

    x_data = np.arange(i * len(actual), i * len(actual) + len(pred))
    ax.plot(x_data, pred, label=f'Forecast {i + 1}')

# 添加文本框展示模型结构
text_box = AnchoredText(
    f"ARIMA Model Structure: ARIMA({best_pdq[0]}, {best_pdq[1]}, {best_pdq[2]})",
    loc='upper left', frameon=True
)
ax.add_artist(text_box)

ax.legend()
ax.set_title("Observed vs Forecasted Crime Counts")
ax.set_xlabel("Time")
ax.set_ylabel("Crime Counts")
plt.show()


# 绘制性能指标箱线图
metrics = [mse_scores, mae_scores, rmse_scores]
metric_names = ['MSE', 'MAE', 'RMSE']
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(metrics, labels=metric_names)
ax.set_title('Distribution of Performance Metrics')
ax.set_ylabel('Metric Value')
plt.show()


# 绘制性能指标折线图
x = range(1, len(mse_scores) + 1)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, mse_scores, label='MSE', marker='o')
ax.plot(x, mae_scores, label='MAE', marker='s')
ax.plot(x, rmse_scores, label='RMSE', marker='^')
ax.set_title('Performance Metrics across Cross - Validation Folds')
ax.set_xlabel('Fold')
ax.set_ylabel('Metric Value')
ax.legend()
plt.show()


# In[5]:


import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
import networkx as nx

#模型结构可视化函数
def visualize_arima_structure(p, d, q):
    G = nx.DiGraph()
    # 添加自回归节点和边
    for i in range(1, p + 1):
        G.add_node(f'AR{i}')
        G.add_edge(f'AR{i}', 'Output')
    # 添加差分节点
    for i in range(1, d + 1):
        G.add_node(f'I{i}')
        if i == 1:
            G.add_edge('Input', f'I{i}')
        else:
            G.add_edge(f'I{i - 1}', f'I{i}')
        G.add_edge(f'I{d}', 'Output')
    # 添加移动平均节点和边
    for i in range(1, q + 1):
        G.add_node(f'MA{i}')
        G.add_edge(f'MA{i}', 'Output')
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10)
    plt.title('ARIMA Model Structure')
    plt.show()
#性能指标可视化函数
def visualize_performance_metrics(mse_scores, mae_scores, rmse_scores):
    x = range(len(mse_scores))
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    # 绘制MSE柱状图
    axs[0].bar(x, mse_scores)
    axs[0].set_title('MSE Scores')
    axs[0].set_xlabel('Fold')
    axs[0].set_ylabel('MSE')
    # 绘制MAE柱状图
    axs[1].bar(x, mae_scores)
    axs[1].set_title('MAE Scores')
    axs[1].set_xlabel('Fold')
    axs[1].set_ylabel('MAE')
    # 绘制RMSE柱状图
    axs[2].bar(x, rmse_scores)
    axs[2].set_title('RMSE Scores')
    axs[2].set_xlabel('Fold')
    axs[2].set_ylabel('RMSE')
    plt.tight_layout()
    plt.show()
# best_pdq已经在之前的代码中确定
p, d, q = best_pdq[0], best_pdq[1], best_pdq[2]
visualize_arima_structure(p, d, q)
visualize_performance_metrics(mse_scores, mae_scores, rmse_scores)
    


# In[ ]:




