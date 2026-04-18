import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.font_manager as font_manager
import matplotlib as mpl
import os

# ================= 1. 全局配置与字体加载 =================
# 配置全局绘图参数 (Nature 风格)
mpl.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',  # 使用 LaTeX 风格数学字体
    'axes.unicode_minus': False,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 300,
})


# === 自动寻找中文字体 ===
def get_chinese_font():
    # 候选字体文件路径 (Linux/Windows 常用路径)
    font_paths = [
        '/usr/share/fonts/truetype/arphic/uming.ttc',
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/msyh.ttc',
        '/System/Library/Fonts/PingFang.ttc'
    ]
    # 1. 尝试加载特定路径字体
    for path in font_paths:
        if os.path.exists(path):
            try:
                prop = font_manager.FontProperties(fname=path, size=14)
                print(f"✅ Loaded font file: {path}")
                return prop
            except:
                continue

    # 2. 尝试加载系统已安装字体名
    candidates = ['AR PL UMing CN', 'SimSun', 'Microsoft YaHei', 'SimHei', 'Songti SC', 'Droid Sans Fallback']
    for family in candidates:
        if family in [f.name for f in font_manager.fontManager.ttflist]:
            print(f"✅ Using installed font family: {family}")
            return font_manager.FontProperties(family=family, size=14)

    # 3. 兜底
    print("⚠️ No specific Chinese font found, using default sans-serif.")
    return font_manager.FontProperties(family='sans-serif', size=14)


chinese_font = get_chinese_font()
roman_font = font_manager.FontProperties(family='DejaVu Serif', size=14)  # 或 Times New Roman

# ================= 2. 数据准备 =================
data = {
    'Coefficient': ['CL', 'CD', 'CY', 'Cl', 'Cm', 'Cn', 'Total'],
    'MSE_No_MTL': [5.6270e-05, 1.1622e-04, 4.4944e-05, 5.6645e-05, 4.4774e-05, 3.1077e-05, 5.8322e-05],
    'MSE_MTL': [1.1024e-05, 2.6501e-05, 1.5927e-05, 1.5937e-05, 1.8561e-05, 9.5916e-06, 1.6167e-05],
    'MAE_No_MTL': [0.005463, 0.000768, 0.001089, 0.000179, 0.001391, 0.000143, 0.001506],
    'MAE_MTL': [0.002223, 0.0003607, 0.0005589, 0.00009190, 0.0007071, 0.00006496, 0.0006678]
}

df = pd.DataFrame(data)
df['MSE_Improvement'] = (df['MSE_No_MTL'] - df['MSE_MTL']) / df['MSE_No_MTL'] * 100

# 标签映射
label_mapping = {
    'CL': {'name': '升力系数', 'math': r'$C_{L}$'},
    'CD': {'name': '阻力系数', 'math': r'$C_{D}$'},
    'CY': {'name': '侧向力系数', 'math': r'$C_{Y}$'},
    'Cl': {'name': '滚转力矩系数', 'math': r'$C_{l}$'},
    'Cm': {'name': '俯仰力矩系数', 'math': r'$C_{m}$'},
    'Cn': {'name': '偏航力矩系数', 'math': r'$C_{n}$'},
    'Total': {'name': '总体平均', 'math': 'Avg'}
}

x_labels = []
for coeff in df['Coefficient']:
    info = label_mapping.get(coeff, {'name': coeff, 'math': coeff})
    x_labels.append(f"{info['name']}\n{info['math']}")

# 配色
COLOR_BASE = '#4C72B0'
COLOR_OURS = '#C44E52'
COLOR_POS = '#55A868'
COLOR_NEG = '#C44E52'

# ================= 3. 绘图部分 =================

# --- 图表 1: MSE 误差对比 ---
plt.figure(figsize=(9, 7))
ax1 = plt.gca()
x = np.arange(len(df['Coefficient']))
width = 0.4

rects1 = ax1.bar(x - width / 2, df['MSE_No_MTL'], width, label='No-STCH',
                 color=COLOR_BASE, alpha=0.9, edgecolor='black', linewidth=0.5)
rects2 = ax1.bar(x + width / 2, df['MSE_MTL'], width, label='STCH',
                 color=COLOR_OURS, alpha=0.9, edgecolor='black', linewidth=0.5)

# 设置标签 (注意这里使用了 chinese_font)
ax1.set_ylabel('均方误差 (MSE)', fontproperties=chinese_font, fontsize=14)
ax1.set_title('单任务与多任务模型误差对比 (MSE)', fontproperties=chinese_font, fontsize=16, pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, fontproperties=chinese_font, fontsize=12)

ax1.set_yscale('log')
ax1.set_ylim(top=df['MSE_No_MTL'].max() * 5)
# ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.legend(prop=chinese_font, loc='upper right', frameon=False)
ax1.set_ylim(top=1.5e-4)

# 添加数值标签
def autolabel_log(rects):
    for rect in rects:
        height = rect.get_height()
        ax1.annotate(f'{height:.1e}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontproperties=roman_font)


autolabel_log(rects1)
autolabel_log(rects2)

plt.tight_layout()
plt.savefig('mse_comparison.png', dpi=300, bbox_inches='tight')
print("Saved mse_comparison.png")
plt.close()  # 关闭当前图表以释放内存

# --- 图表 2: 性能提升百分比 ---
plt.figure(figsize=(9, 7))
ax2 = plt.gca()

colors = [COLOR_POS if val > 0 else COLOR_NEG for val in df['MSE_Improvement']]
rects3 = ax2.bar(x, df['MSE_Improvement'], width=0.6, color=colors,
                 alpha=0.85, edgecolor='black', linewidth=0.5)

# 设置标签
ax2.set_ylabel('性能提升百分比 (%)', fontproperties=chinese_font, fontsize=14)
ax2.set_title('多任务学习相对性能提升率', fontproperties=chinese_font, fontsize=16, pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels, fontproperties=chinese_font, fontsize=12)

ax2.set_ylim(0, df['MSE_Improvement'].max() * 1.15)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.axhline(y=0, color='black', linewidth=1)

# 添加数值
for rect in rects3:
    height = rect.get_height()
    ax2.text(rect.get_x() + rect.get_width() / 2., height + 1,
             f'{height:.1f}%',
             ha='center', va='bottom', fontweight='bold',
             fontsize=10, fontproperties=roman_font)

plt.tight_layout()
plt.savefig('improvement_comparison.png', dpi=300, bbox_inches='tight')
print("Saved improvement_comparison.png")
plt.close()