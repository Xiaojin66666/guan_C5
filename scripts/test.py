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
x = torch.load("/home/user/ftd_project 2/saved_x4.pt", map_location=torch.device("cpu")).float().numpy()
y = torch.load("/home/user/ftd_project 2/saved_y7.pth", map_location=torch.device("cpu")).float().numpy()
preds = torch.load("/home/user/ftd_project 2/saved_preds7.pth", map_location=torch.device("cpu")).float().numpy()

min_data = np.load("/home/user/ftd_project 2/output_min1.npy")
max_data = np.load("/home/user/ftd_project 2/output_max1.npy")

y[0] = y[0] * (max_data - min_data) + min_data
preds = preds * (max_data - min_data) + min_data

title = ["$C_{Y}$", "$C_{l}$", "$C_{m}$", "$C_{n}$", "$C_{D}$", "$C_{L}$"]

for i in range(6):
    # 排序 x[0, :, 0] 并获取排序的索引
    sorted_indices = np.argsort(x[0, :2561, 0])

    # 使用排序索引对 x, y, preds 进行排序
    sorted_x = x[0, :2561, 0][sorted_indices]
    sorted_y = y[0, :2561, i][sorted_indices]
    sorted_preds = preds[:2561, i][sorted_indices]

    plt.style.use(['science', 'no-latex'])
    mpl.rcParams['axes.linewidth'] = 1.0

    plt.figure(figsize=(7, 5), dpi=200)
    # plt.title(title[i])
    plt.plot(sorted_x * 9300, sorted_y, color="black", label="Real")
    plt.plot(sorted_x * 9300, sorted_preds, color="red", label="Pred")

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

    plt.show()
