import matplotlib.pyplot as plt
import numpy as np
import torch
import scienceplots
import matplotlib.font_manager as font_manager
import matplotlib as mpl
import matplotlib.ticker as ticker

font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=20)
x = np.linspace(-4, 4, 400)

# Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU函数
def relu(x):
    return np.where(x < 0, 0, x)

# Tanh函数
def tanh(x):
    return np.tanh(x)

# 计算各激活函数的值
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)

# 绘图
plt.style.use(['science', 'no-latex'])
mpl.rcParams['axes.linewidth'] = 1.0
plt.figure(figsize=(7, 5),dpi=200)
plt.plot(x, y_sigmoid, label='sigmoid',color="black")

plt.plot(x, y_tanh, label='tanh',color="g")
plt.plot(x, y_relu, label='ReLU',color="red")
# 设置x轴标签字体
plt.xlabel('x', fontsize=20, fontproperties=font_prop)
# 设置y轴标签字体
plt.ylabel('y', fontsize=20, fontproperties=font_prop)

# 设置图例字体
plt.legend(fontsize=20, prop=font_prop)
plt.tick_params(axis='x', which='major', labelsize=20)
plt.tick_params(axis='y', which='major', labelsize=20)

plt.show()
