#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os

path = r'C:\Users\86173\Downloads\sf-crime\train.csv'  
os.chdir(path)

df= pd.read_csv('train6.csv', encoding='gbk')
pd.set_option('display.max_columns', None)
# 获取列数
num_cols = df.shape[1]
# 去除最后一列（索引为num_cols - 1）
df = df.drop(df.columns[num_cols - 1], axis=1)
print(df.head())
print(df.shape)


# In[2]:


sample_data = pd.read_csv('train.csv', nrows=10)
print(sample_data.dtypes)


# In[3]:


print(df.isnull().sum())


# In[4]:


df = df.dropna(axis=0, how='any')
print(df.shape)


# In[5]:


#绘制犯罪类别直方图（category）
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
category_counts = df['Category'].value_counts()
plt.figure(figsize=(12, 8))
plt.bar(category_counts.index, category_counts.values)
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Distribution of Category')
plt.xticks(rotation=90)
plt.show()


# In[6]:


usecols = ['Dates', 'X','Y', 'Category']
df.shape, df.drop_duplicates(subset=usecols).shape


# In[7]:


print(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)


# In[8]:


df['X'].describe()


# In[9]:


df['Y'] = pd.to_numeric(df['Y'], errors='coerce')


# In[10]:


df['Y'].describe()


# In[11]:


df[df.Y > 38].shape


# In[12]:


df[df.Y > 38].duplicated(subset=['X', 'Y', 'Category'], keep=False).sum()


# In[13]:


df = df[df.Y <= 38].reset_index(drop=True)


# In[14]:


#犯罪类别在地理空间的分布
import seaborn as sns
sns.scatterplot(
    data=df.drop_duplicates(subset=['X','Y', 'Category']), 
    x='X', 
    y='Y', 
    hue='Category'
)
plt.legend(labels = []);


# In[15]:


#不同警区的犯罪集中区域
sns.scatterplot(
    data=df.drop_duplicates(subset=['X','Y', 'Category', 'PdDistrict']), 
    x='X', 
    y='Y', 
    hue='PdDistrict'
)
plt.legend(labels = []);


# In[16]:


print('数据集开始日期: ', df.Dates.min())
print('数据集结束日期: ', df.Dates.max())


# In[17]:


#绘制每天发生的犯罪事件数量的核密度估计图，并绘制中位数的垂直线
# 将Dates列转换为日期时间类型
df['Dates'] = pd.to_datetime(df['Dates'])
# 提取日期部分，并按照日期分组统计每天的犯罪事件数量
daily_counts = df['Dates'].dt.date.groupby(df['Dates'].dt.date).count()
# 绘制核密度估计图
sns.kdeplot(data=daily_counts, fill=True)
# 计算中位数
median_daily = daily_counts.median()
# 绘制中位数的垂直线
plt.axvline(median_daily, color='r', linestyle='--', label='Median')
# 设置图形的标题、坐标轴标签等
plt.title('Daily Crime Counts Kernel Density Estimation')
plt.xlabel('Number of Crimes')
plt.ylabel('Density')
# 显示图例
plt.legend()
# 显示图形
plt.show()


# In[18]:


#绘制所有天数内每个时段发生的犯罪事件数量的核密度估计图，并绘制中位数的垂直线
# 提取小时部分，作为时段的代表（这里简单以小时划分时段，你也可以根据需要细化，比如每半小时等）
df['Hour'] = df['Dates'].dt.hour
# 按照小时分组统计每个时段的犯罪事件数量
hourly_counts = df['Hour'].groupby(df['Hour']).count()
# 绘制核密度估计图
sns.kdeplot(data=hourly_counts, fill=True)
# 计算中位数
median_hourly = hourly_counts.median()
# 绘制中位数的垂直线
plt.axvline(median_hourly, color='r', linestyle='--', label='Median')
# 设置图形的标题、坐标轴标签等
plt.title('Hourly Crime Counts Kernel Density Estimation')
plt.xlabel('Number of Crimes')
plt.ylabel('Density')
# 显示图例
plt.legend()
# 显示图形
plt.show()


# In[19]:


from wordcloud import WordCloud
import numpy as np


# In[20]:


#绘制 5 个犯罪类型的平均每小时事件数折线图
df['Dates'] = pd.to_datetime(df['Dates'])

# 提取小时信息，创建新列 'Hour'，表示每个事件发生的小时数
df['Hour'] = df['Dates'].dt.hour

# 按照犯罪类型和小时进行分组，统计每个分组中的事件数量（比如每小时每种犯罪类型发生了多少起事件）
HourCategory = df.groupby(['Category', 'Hour']).size().reset_index(name='Incidents')

# 后续就是你提供的代码部分，用于绘制折线图
Category_5 = ['LARCENY/THEFT', 'NON - CRIMINAL', 'OTHER OFFENSES', 'ASSAULT', 'VANDALISM']
HourCategory_5 = HourCategory.loc[HourCategory['Category'].isin(Category_5)]
fig, ax = plt.subplots(figsize=(14, 6))
ax = sns.lineplot(x='Hour', y='Incidents', data=HourCategory_5,
                  hue='Category', hue_order=Category_5, style="Category", markers=True, dashes=False)
ax.legend(loc='upper center', ncol=5)
plt.suptitle('一天中每个时间段每种犯罪类型的平均事件数折线图')
fig.tight_layout()
plt.show()


# In[21]:


#绘制不同星期几下的犯罪事件数量占比
# 按星期几分组，计算每组的事件数
day_of_week_counts = df.groupby('DayOfWeek').size()

# 计算占比 
day_of_week_percentage = day_of_week_counts / day_of_week_counts.sum() * 100

# 绘制柱状图
plt.bar(day_of_week_percentage.index, day_of_week_percentage)
plt.xlabel('Day of Week')
plt.ylabel('Percentage of Crime Events')
plt.title('Percentage of Crime Events by Day of Week')
plt.show()


# In[22]:


df['Dates'] = pd.to_datetime(df['Dates'])
# df是包含犯罪数据的数据框，且Dates列已转换为日期时间类型
# 按警察区和日期分组，计算每组的事件数
district_daily_counts = df.groupby(['PdDistrict', df['Dates'].dt.date]).size().unstack()

# 计算平均每天的犯罪频率
average_daily_frequency = district_daily_counts.mean(axis = 0)
print(average_daily_frequency)


# In[23]:


#计算每一天中发生的犯罪频率并绘图
# 按警察区和日期分组，计算每组的事件数
district_daily_counts = df.groupby(['PdDistrict', df['Dates'].dt.date]).size().unstack()
# 计算平均每天的犯罪频率
average_daily_frequency = district_daily_counts.mean()

# 绘制柱状图
plt.bar(average_daily_frequency.index, average_daily_frequency)
plt.xlabel('Police District')
plt.ylabel('Average Daily Crime Frequency')
plt.title('Average Daily Crime Frequency by Police District')
plt.xticks(rotation=90)
plt.show()



# In[24]:


#绘制不同犯罪类型的地理密度图

# 选择需要的列
geo_df = df[['X', 'Y', 'Category']]
geo_df['X'] = pd.to_numeric(geo_df['X'], errors='coerce')
geo_df['Y'] = pd.to_numeric(geo_df['Y'], errors='coerce')
# 绘制地理密度图
for crime_type in df['Category'].unique():
    subset = geo_df[geo_df['Category'] == crime_type]
    # 创建一个临时的数据框，明确指定X和Y列的数据类型为数值型
    subset_plot = pd.DataFrame({
        'X': subset['X'].astype(float),
        'Y': subset['Y'].astype(float)
    })
    sns.kdeplot(data = subset, x = 'X', y = 'Y', label = crime_type)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographical Density of Different Crime Types')
plt.legend()

plt.figure(figsize=(20, 10))
plt.show()


# In[25]:


df.Descript.nunique(), df.PdDistrict.nunique(), df.Address.nunique()


# In[26]:


#计算每个警察区区域平均一天中发生的犯罪频率并绘图
import pandas as pd
# df是包含犯罪数据的数据框，且Dates列已转换为日期时间类型
df['Dates'] = pd.to_datetime(df['Dates'])
# 按警察区和日期分组，计算每组的事件数
district_daily_counts = df.groupby(['PdDistrict', df['Dates'].dt.date]).size().unstack()
# 计算平均每天的犯罪频率（按警区）
average_daily_Crime_frequency = district_daily_counts.mean(axis = 1)
print(average_daily_Crime_frequency)


# In[27]:


# 绘制柱状图
plt.bar(average_daily_Crime_frequency.index, average_daily_Crime_frequency)
plt.xlabel('Police District')
plt.ylabel('Average Daily Crime Frequency')
plt.title('Average Daily Crime Frequency by Police District')
plt.xticks(rotation=90)
plt.show()


# In[ ]:




