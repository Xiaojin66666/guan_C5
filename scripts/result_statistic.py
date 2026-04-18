import matplotlib.pyplot as plt
import numpy as np
import torch
import scienceplots
import matplotlib.font_manager as font_manager
import matplotlib as mpl
import matplotlib.ticker as ticker

# ================= 字体设置 =================
# 设置中文字体（宋体）和英文字体（Times New Roman）
chinese_font_path = '/usr/share/fonts/chinese/ARPLUMingCN.ttf'  # 宋体字体路径，请根据实际情况修改
english_font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'

# 尝试加载宋体
try:
    if font_manager.findfont('AR PL UMing CN'):
        chinese_font_prop = font_manager.FontProperties(family='AR PL UMing CN', size=15)
        print("✓ 成功加载宋体字体: AR PL UMing CN")
    else:
        chinese_font_prop = font_manager.FontProperties(family='AR PL UMing CN', size=15)
except:
    # 如果找不到宋体，使用默认字体
    chinese_font_prop = font_manager.FontProperties(family='sans-serif', size=15)
    print("⚠ 使用默认字体代替宋体")

# 尝试加载Times New Roman
try:
    if font_manager.findfont('Times New Roman'):
        english_font_prop = font_manager.FontProperties(family='Times New Roman', size=15)
        print("✓ 成功加载Times New Roman字体")
    else:
        english_font_prop = font_manager.FontProperties(family='Times New Roman', size=15)
except:
    english_font_prop = font_manager.FontProperties(family='serif', size=15)
    print("⚠ 使用默认字体代替Times New Roman")

# 设置全局字体回退顺序
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================= 加载数据 =================
x = np.load('/home/user/bishe/ftd_lstm/scripts/x_sorted_28bcc4427e9c4909a255c300.npy')
y = np.load('/home/user/bishe/ftd_lstm/scripts/y_sorted_28bcc4427e9c4909a255c300.npy')

print(f"数据形状: x={x.shape}, y={y.shape}")
print(f"x范围: {np.min(x):.2f} 到 {np.max(x):.2f}")
print(f"y范围: {np.min(y):.6f} 到 {np.max(y):.6f}")

# ================= 绘图设置 =================
plt.style.use(['science', 'no-latex'])
mpl.rcParams['axes.linewidth'] = 1.0

# 创建图形
fig = plt.figure(figsize=(7, 5), dpi=300)
ax = plt.gca()

# ================= 绘制图形 =================
# 使用 semilogy 绘制对数刻度的y轴，并添加标签
line, = ax.plot(x[:,], y[:,], color="black", linewidth=1.5, label="验证集损失")

# ================= 设置坐标轴标签 =================
# x轴标签使用宋体
ax.set_xlabel("迭代轮次", fontproperties=chinese_font_prop)
# y轴标签使用宋体
ax.set_ylabel("验证集损失", fontproperties=chinese_font_prop)

# ================= 设置刻度字体 =================
# x轴刻度标签使用Times New Roman
for label in ax.get_xticklabels():
    label.set_fontproperties(english_font_prop)

# y轴刻度标签使用Times New Roman
for label in ax.get_yticklabels():
    label.set_fontproperties(english_font_prop)

# ================= 设置刻度参数 =================
ax.tick_params(axis='x', which='major', labelsize=15)
ax.tick_params(axis='y', which='major', labelsize=15)

# ================= 添加图例 =================
# 图例使用宋体
ax.legend(prop=chinese_font_prop, loc='upper right', frameon=False)


# ================= 调整布局并显示 =================
plt.tight_layout()

# ================= 保存图形 =================
# 可以选择保存图形
save_path = "validation_loss_log_plot.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✓ 图形已保存为: {save_path}")

plt.show()
