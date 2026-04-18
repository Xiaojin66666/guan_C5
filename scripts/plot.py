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


x = torch.load("/home/user/bishe/ftd_lstm/saved_x.pth", map_location=torch.device("cpu")).float().numpy()
y = torch.load("/home/user/bishe/ftd_lstm/saved_y.pth", map_location=torch.device("cpu")).float().numpy()
preds = torch.load("/home/user/bishe/ftd_lstm/saved_preds.pth", map_location=torch.device("cpu")).float().numpy()
print(x.shape)
print(y.shape)
print(preds.shape)

imput_min_data = np.load("/home/user/bishe/ftd_lstm/input_scaler_min.npy")
imput_max_data = np.load("/home/user/bishe/ftd_lstm/input_scaler_max.npy")

output_min_data = np.load("/home/user/bishe/ftd_lstm/output_scaler_min.npy")
output_max_data = np.load("/home/user/bishe/ftd_lstm/output_scaler_max.npy")

x= x* (imput_max_data - imput_min_data) + imput_min_data
y = y* (output_max_data - output_min_data) + output_min_data
preds= preds* (output_max_data - output_min_data) + output_min_data

# ---------------------------------------------------------
# [新增代码] 在画图前进行排序
# 假设 x[:, 0, 0] 是时间轴 t/s，我们需要根据它来排序
# ---------------------------------------------------------
sort_indices = np.argsort(x[:, 0, 0])  # 获取按时间排序的索引
x = x[sort_indices]  # 对 x 进行重排
y = y[sort_indices]  # 对 y 进行同步重排
preds = preds[sort_indices]  # 对 preds 进行同步重排
# ---------------------------------------------------------
title = ["$C_{Y}$", "$C_{l}$", "$C_{m}$", "$C_{n}$", "$C_{D}$", "$C_{L}$"]

for i in range(6):
    # 排序 x[0, :, 0] 并获取排序的索引

    plt.style.use(['science', 'no-latex'])
    mpl.rcParams['axes.linewidth'] = 1.0

    plt.figure(figsize=(7, 5), dpi=200)
    # plt.title(title[i])
    plt.plot(x[:, 0, 0] , y[:, 0, i], color="black", label="Real")
    plt.plot(x[:, 0, 0] , preds[:, 0, i], color="red", label="Pred")
    print(f"i={i}, preds min={np.min(y[:, 0, i])}, max={np.max(y[:, 0, i])}")
    print(f"i={i}, preds min={np.min(preds[:, 0, i])}, max={np.max(preds[:, 0, i])}")

    # 设置x轴标签字体
    plt.xlabel("t/s", fontsize=20, fontproperties=font_prop)

    # 临时启用LaTeX渲染ylabel
    with plt.rc_context({'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Times New Roman']}):
        plt.ylabel(title[i], fontsize=20)

    # 只需要设置一次图例
    plt.legend(prop = font_prop)

    # 设置x轴刻度字体
    for label in plt.gca().get_xticklabels():
        label.set_fontproperties(font_prop)
    # 设置y轴刻度字体
    for label in plt.gca().get_yticklabels():
        label.set_fontproperties(font_prop)

    plt.tick_params(axis='x', which='major', labelsize=20)
    plt.tick_params(axis='y', which='major', labelsize=20)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()
    plt.show()
