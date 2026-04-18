import matplotlib.pyplot as plt
import numpy as np
import torch
import scienceplots
import matplotlib.font_manager as font_manager
import matplotlib as mpl
import matplotlib.ticker as ticker

# 设置字体为Times New Roman
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=20)

# 加载数据
x = torch.load("/home/user/bishe/ftd_lstm/saved_x.pth", map_location=torch.device("cpu")).float().numpy()
y = torch.load("/home/user/bishe/ftd_lstm/saved_y.pth", map_location=torch.device("cpu")).float().numpy()
preds = torch.load("/home/user/bishe/ftd_lstm/saved_preds.pth", map_location=torch.device("cpu")).float().numpy()

imput_min_data = np.load("/home/user/bishe/ftd_lstm/input_scaler_min.npy")
imput_max_data = np.load("/home/user/bishe/ftd_lstm/input_scaler_max.npy")

output_min_data = np.load("/home/user/bishe/ftd_lstm/output_scaler_min.npy")
output_max_data = np.load("/home/user/bishe/ftd_lstm/output_scaler_max.npy")

# 只反归一化时间特征（第一个特征）
time_feature_idx = 2  # 时间特征是第一个特征
time_min = imput_min_data[time_feature_idx]
time_max = imput_max_data[time_feature_idx]

# 提取所有样本的时间特征（第一个时间步的第一个特征）
time_values = x[0, :, 0, 0]  # 形状 (7031,)
original_time = time_values * (time_max - time_min) + time_min

# 输出反归一化
y[0] = y[0] * (output_max_data - output_min_data) + output_min_data
preds = preds * (output_max_data - output_min_data) + output_min_data

title = ["$C_{Y}$", "$C_{l}$", "$C_{m}$", "$C_{n}$", "$C_{D}$", "$C_{L}$"]

for i in range(6):
    # 按时间排序
    sorted_indices = np.argsort(original_time)
    sorted_time = original_time[sorted_indices]
    sorted_y = y[0, :, i][sorted_indices]
    sorted_preds = preds[:, i][sorted_indices]

    plt.style.use(['science', 'no-latex'])
    mpl.rcParams['axes.linewidth'] = 1.0

    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(sorted_time, sorted_y, color="black", label="Real")
    plt.plot(sorted_time, sorted_preds, color="red", label="Pred")

    plt.xlabel("t/s", fontsize=20, fontproperties=font_prop)

    # 使用LaTeX渲染ylabel
    with plt.rc_context({'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Times New Roman']}):
        plt.ylabel(title[i], fontsize=20)

    plt.legend(prop=font_prop)

    # 设置坐标轴字体
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)

    plt.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    plt.tight_layout()
    #plt.savefig(f"plot_{title[i]}.png", dpi=300, bbox_inches='tight')
    plt.show()