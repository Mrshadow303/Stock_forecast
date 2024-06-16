#此文件是为了可视化初始数据用的，与训练无关

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
file_path = 'stock_dataset_2.csv'  
df = pd.read_csv(file_path)

# 将日期列转换为datetime格式
df['date'] = pd.to_datetime(df['date'])

# 设置日期列为索引
df.set_index('date', inplace=True)

# 绘制开盘价、收盘价、最低价、最高价的时间序列图
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['open'], label='Open Price')
plt.plot(df.index, df['close'], label='Close Price')
plt.plot(df.index, df['low'], label='Low Price')
plt.plot(df.index, df['high'], label='High Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Prices Over Time')
plt.legend()
plt.show()

# 绘制成交量的时间序列图
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['volume'], label='Volume', color='purple')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Trading Volume Over Time')
plt.legend()
plt.show()

# 绘制涨幅的时间序列图
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['change'], label='Change', color='green')
plt.xlabel('Date')
plt.ylabel('Change')
plt.title('Stock Price Change Over Time')
plt.legend()
plt.show()

# 使用seaborn绘制开盘价与收盘价的关系图
plt.figure(figsize=(14, 7))
sns.scatterplot(x='open', y='close', data=df)
plt.xlabel('Open Price')
plt.ylabel('Close Price')
plt.title('Open Price vs Close Price')
plt.show()

# 使用seaborn绘制成交量与涨幅的关系图
plt.figure(figsize=(14, 7))
sns.scatterplot(x='volume', y='change', data=df)
plt.xlabel('Volume')
plt.ylabel('Change')
plt.title('Volume vs Change')
plt.show()