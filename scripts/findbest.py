import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# ====================
# 字体设置：分离中英文字体
# ====================
# 创建字体属性对象
chinese_font_prop = FontProperties(family='AR PL UMing CN', size=12)
english_font_prop = FontProperties(family='Times New Roman', size=11)

# 设置全局字体 - 优先使用英文字体
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'AR PL UMing CN']
plt.rcParams['axes.unicode_minus'] = False

# ====================
# 数据准备
# ====================
df_depth = pd.DataFrame({
    'Layers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'MSE': [12.87, 5.41, 3.48, 2.52, 2.23, 1.90, 1.48, 2.12, 1.66, 2.78, 1.84, 1.62, 2.36, 1.76],
    'Time': [6.3, 8.0, 9.5, 10.8, 13.0, 14.8, 14.5, 10.8, 17.5, 8.6, 14.7, 18.1, 13.9, 21.9]
})

df_neuron = pd.DataFrame({
    'Neurons': ['3+0', '2+1', '1+2', '4+0', '3+1', '2+2'],
    'MSE': [2.153, 1.8, 1.744, 1.981, 1.868, 1.827],
    'Time': [115, 145, 135, 130, 154, 138]
})

grid_data = np.array([
    [2.967, 2.161, 1.164, 0.9173],
    [2.761, 1.438, 1.541, 0.8026],
    [1.822, 3.905, 4.228, 0.4286],
    [2.356, 4.822, 4.149, 0.5775],
])
df_grid = pd.DataFrame(grid_data, index=[2, 3, 4, 5], columns=[32, 64, 96, 128])

df_lr = pd.DataFrame({
    'LR': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    'MSE': [2.892, 1.725, 1.6, 573, 1.674, 573.7, 64.5],
    'Time': [136, 135, 137, 17.75, 80, 10, 18.67]
})

# ====================
# 图形1：网络深度敏感性分析
# ====================
fig1 = plt.figure(figsize=(7, 5))  # 7:5比例
ax1 = fig1.add_subplot(111)
ax1_twin = ax1.twinx()

# 禁用seaborn自动生成的图例
sns.lineplot(x='Layers', y='MSE', data=df_depth, ax=ax1, marker='o', color='b',
             linewidth=2.5, legend=False)
sns.lineplot(x='Layers', y='Time', data=df_depth, ax=ax1_twin, marker='s', color='r',
             linestyle='--', linewidth=2, legend=False)

# 设置字体：中文用宋体，英文和数字用罗马体
ax1.set_xlabel('隐藏层数量', fontproperties=chinese_font_prop)
ax1.set_ylabel('验证集均方误差 (×10$^{-5}$)', color='b', fontproperties=chinese_font_prop)
ax1_twin.set_ylabel('训练时间 (分钟)', color='r', fontproperties=chinese_font_prop)
ax1.set_title('(a) 网络深度敏感性分析', fontproperties=chinese_font_prop, fontweight='bold', size=14)
ax1.grid(True, linestyle='--', alpha=0.6)

# 设置刻度标签字体
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontproperties(english_font_prop)
for label in ax1_twin.get_yticklabels():
    label.set_fontproperties(english_font_prop)

# 手动创建图例
from matplotlib.lines import Line2D
legend_lines = [
    Line2D([0], [0], color='b', marker='o', linestyle='-', linewidth=2.5, label='均方误差 (×10$^{-4}$)'),
    Line2D([0], [0], color='r', marker='s', linestyle='--', linewidth=2, label='训练时间 (分钟)')
]
ax1.legend(handles=legend_lines, loc='upper center', prop=chinese_font_prop, frameon=True)

plt.tight_layout()
plt.savefig('figure1_depth_analysis.png', dpi=500, bbox_inches='tight')
plt.show()

# ====================
# 图形2：网络宽度敏感性分析
# ====================
fig2 = plt.figure(figsize=(7, 5))  # 7:5比例
ax2 = fig2.add_subplot(111)
ax2_twin = ax2.twinx()

# 柱状图画时间
bars = ax2_twin.bar(df_neuron['Neurons'], df_neuron['Time'], color='r', alpha=0.3)
# 折线图画误差
line, = ax2.plot(df_neuron['Neurons'], df_neuron['MSE'], marker='o', color='b',
                 linewidth=2.5)

ax2.set_xlabel('共享层+特定层数量', fontproperties=chinese_font_prop)
ax2.set_ylabel('验证集均方误差 (×10$^{-5}$)', color='b', fontproperties=chinese_font_prop)
ax2_twin.set_ylabel('训练时间 (分钟)', color='r', fontproperties=chinese_font_prop)
ax2.set_title('(b) 网络宽度敏感性分析', fontproperties=chinese_font_prop, fontweight='bold', size=14)

# 添加横刻度虚线网格
ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
# 设置刻度标签字体
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontproperties(english_font_prop)
for label in ax2_twin.get_yticklabels():
    label.set_fontproperties(english_font_prop)

# 自定义图例
legend_elements = [
    Line2D([0], [0], color='b', lw=2.5, marker='o', label='均方误差 (×10$^{-5}$)'),
    mpl.patches.Patch(facecolor='r', alpha=0.3, label='训练时间 (分钟)')
]
ax2.legend(handles=legend_elements, loc='upper center', prop=chinese_font_prop, frameon=True)

plt.tight_layout()
plt.savefig('figure2_neuron_analysis.png', dpi=500, bbox_inches='tight')
plt.show()

# ====================
# 图形3：局部网格搜索
# ====================
fig3 = plt.figure(figsize=(7, 5))  # 7:5比例
ax3 = fig3.add_subplot(111)

# annot_kws 设置数字字体
heatmap = sns.heatmap(df_grid, annot=True, fmt=".2f", cmap="YlGnBu_r",
                      cbar_kws={'label': '均方误差 (×10$^{-4}$)'}, ax=ax3,
                      annot_kws={"size": 12, "fontproperties": english_font_prop})

# 设置热力图颜色条标签字体
cbar = heatmap.collections[0].colorbar
cbar.ax.set_ylabel('均方误差 (×10$^{-4}$)', fontproperties=chinese_font_prop)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(english_font_prop)

ax3.set_xlabel('神经元数量', fontproperties=chinese_font_prop)
ax3.set_ylabel('隐藏层数量', fontproperties=chinese_font_prop)
ax3.set_title('(c) 最优架构局部网格搜索', fontproperties=chinese_font_prop, fontweight='bold', size=14)

# 设置坐标轴标签字体
for label in ax3.get_xticklabels() + ax3.get_yticklabels():
    label.set_fontproperties(english_font_prop)

plt.tight_layout()
plt.savefig('figure3_grid_search.png', dpi=500, bbox_inches='tight')
plt.show()

# ====================
# 图形4：学习率影响
# ====================
fig4 = plt.figure(figsize=(7, 5))  # 7:5比例
ax4 = fig4.add_subplot(111)
ax4_twin = ax4.twinx()

# 禁用seaborn自动生成的图例
sns.lineplot(x='LR', y='MSE', data=df_lr, ax=ax4, marker='o', color='g',
             linewidth=2.5, legend=False)
ax4_twin.plot(df_lr['LR'], df_lr['Time'], 'm--s')

ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.set_xlabel('初始学习率 (对数坐标)', fontproperties=chinese_font_prop)
ax4.set_ylabel('验证集均方误差 (×10$^{-5}$)', color='g', fontproperties=chinese_font_prop)
ax4_twin.set_ylabel('训练时间 (分钟)', color='m', fontproperties=chinese_font_prop)
ax4.set_title('(d) 初始学习率对收敛特性的影响', fontproperties=chinese_font_prop, fontweight='bold', size=14)
ax4.grid(True, which="both", ls="--", alpha=0.5)

# 设置刻度标签字体
for label in ax4.get_xticklabels() + ax4.get_yticklabels():
    label.set_fontproperties(english_font_prop)
for label in ax4_twin.get_yticklabels():
    label.set_fontproperties(english_font_prop)

# 创建图例
legend_lines2 = [
    Line2D([0], [0], color='g', marker='o', linestyle='-', linewidth=2.5, label='均方误差 (×10$^{-5}$)'),
    Line2D([0], [0], color='m', marker='s', linestyle='--', linewidth=2, label='训练时间 (分钟)')
]
ax4.legend(handles=legend_lines2, loc='center left', prop=chinese_font_prop, frameon=True)

plt.tight_layout()
plt.savefig('figure4_learning_rate.png', dpi=500, bbox_inches='tight')
plt.show()

print("已生成4个单独的图形文件:")
print("1. figure1_depth_analysis.png - 网络深度敏感性分析")
print("2. figure2_neuron_analysis.png - 网络宽度敏感性分析（已添加横刻度虚线）")
print("3. figure3_grid_search.png - 局部网格搜索")
print("4. figure4_learning_rate.png - 学习率影响分析")
print("所有图形都已保存为7:5比例、500ppi的高分辨率图像。")