import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 读取CSV文件
file_path = 'stock_dataset_2.csv'  
df = pd.read_csv(file_path)

# 将日期列转换为datetime格式
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 选择特征和目标
# features = ['open', 'close', 'low', 'high', 'volume', 'money', 'change']
features = ['open', 'close']
target = 'label'

epochs = 500                    #训练轮数
time_step = 120                 #步长
# 检查是否继续训练
continue_training = True        #决定是否继续训练
model_path = 'Stock_0.0182.h5'  #之前保存的模型文件路径

# 缩放数据
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaler_label = MinMaxScaler(feature_range=(0, 1))

scaled_features = scaler_features.fit_transform(df[features])
scaled_label = scaler_label.fit_transform(df[[target]])

scaled_data = np.hstack((scaled_features, scaled_label))

# 准备训练数据
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :-1])  # 排除最后一列
        Y.append(data[i + time_step, -1])       # 最后一列作为标签
    return np.array(X), np.array(Y)


X, Y = create_dataset(scaled_data, time_step)

# 将数据集分为训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# 如果选择继续训练，则加载最优模型，否则构建新的模型
if continue_training:
    # 重新定义模型
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # 编译模型
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # 加载最优模型权重
    model.load_weights(model_path)
else:
    # 构建新的模型
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # 编译模型
    model.compile(optimizer='sgd', loss='mean_squared_error')

# 定义一个自定义回调来根据验证RMSE保存模型
class CustomModelCheckpoint(Callback):
    def __init__(self, monitor='val_loss', verbose=0, save_best_only=True, mode='min'):
        super(CustomModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = np.Inf if mode == 'min' else -np.Inf
        self.best_filepath = ""

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.mode == 'min' and current < self.best:
            self.best = current
            rmse = np.sqrt(current)
            self.best_filepath = f'Stock_{rmse:.4f}.h5'
            self.model.save(self.best_filepath)
            if self.verbose > 0:
                print(f'\nEpoch {epoch + 1}: saving model to {self.best_filepath}')
        elif self.mode == 'max' and current > self.best:
            self.best = current
            rmse = np.sqrt(current)
            self.best_filepath = f'Stock_{rmse:.4f}.h5'
            self.model.save(self.best_filepath)
            if self.verbose > 0:
                print(f'\nEpoch {epoch + 1}: saving model to {self.best_filepath}')

# 配置回调函数，包括早停和模型检查点
checkpoint = CustomModelCheckpoint(monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 训练模型并使用回调函数
model.fit(X_train, Y_train, batch_size=32, epochs=epochs, validation_data=(X_test, Y_test), verbose=1, callbacks=[checkpoint, early_stopping])

# 进行预测
Y_pred = model.predict(X_test)

# 反归一化预测结果
Y_test_rescaled = scaler_label.inverse_transform(Y_test.reshape(-1, 1)).flatten()
Y_pred_rescaled = scaler_label.inverse_transform(Y_pred).flatten()

# 计算归一化和反归一化的均方根误差
mse_norm = mean_squared_error(Y_test, Y_pred)
rmse_norm = np.sqrt(mse_norm)

mse_orig = mean_squared_error(Y_test_rescaled, Y_pred_rescaled)
rmse_orig = np.sqrt(mse_orig)

# 输出结果
print(f'Normalized Mean Squared Error (归一化均方误差, MSE): {mse_norm}')
print(f'Normalized Root Mean Squared Error (归一化均方根误差, RMSE): {rmse_norm}')

print(f'Original Mean Squared Error (均方误差, MSE): {mse_orig}')
print(f'Original Root Mean Squared Error (均方根误差, RMSE): {rmse_orig}')

# 可视化预测结果
plt.figure(figsize=(14, 7))
plt.plot(Y_test_rescaled, label='Actual Label')
plt.plot(Y_pred_rescaled, label='Predicted Label')
plt.xlabel('Samples')
plt.ylabel('Label')
plt.title('Actual vs Predicted Label')
plt.legend()
plt.show()
