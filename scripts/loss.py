import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib as mpl
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from aim import Repo

# ================= 🔧 用户配置区 =================

# 1. 您的目标 Run Hash
TARGET_HASH = "28bcc4427e9c4909a255c300"

# 2. 目标指标 (验证集 Loss)
METRIC_NAME = "loss"
SUBSET_LABEL = "val"

# 3. 图片保存名称
OUTPUT_FILENAME = f"raw_curve_sorted_{TARGET_HASH}.png"

# 4. 数据保存名称
X_DATA_FILENAME = f"x_sorted_{TARGET_HASH}.npy"
Y_DATA_FILENAME = f"y_sorted_{TARGET_HASH}.npy"
# ===============================================

def find_aim_repo(start_path):
    """自动向上查找 .aim 文件夹"""
    current_path = os.path.abspath(start_path)
    while True:
        aim_path = os.path.join(current_path, ".aim")
        if os.path.exists(aim_path) and os.path.isdir(aim_path):
            return current_path
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            return None
        current_path = parent_path


def set_nature_style():
    """配置 Nature 顶刊绘图风格"""
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    font_prop = None
    try:
        if os.path.exists(font_path):
            font_prop = font_manager.FontProperties(fname=font_path, size=15)
            plt.rcParams['font.family'] = 'serif'
        else:
            font_prop = font_manager.FontProperties(family='serif', size=15)
    except:
        pass

    mpl.rcParams.update({
        'font.size': 16,
        'axes.linewidth': 1.0,
        'axes.unicode_minus': False,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'legend.frameon': False,
        'figure.dpi': 300,
    })
    return font_prop


def main():
    # 1. 连接 Aim
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_path = find_aim_repo(script_dir)
    if not repo_path:
        print("❌ Error: Could not find '.aim' repository.")
        return
    print(f"📂 Connected to Aim Repo: {repo_path}")

    # 2. 获取 Run
    try:
        repo = Repo(repo_path)
        run = repo.get_run(TARGET_HASH)
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return

    if not run:
        print(f"❌ Run '{TARGET_HASH}' not found.")
        return

    # 3. 查找指标
    print(f"🔍 Searching for metric: '{METRIC_NAME}' (subset='{SUBSET_LABEL}')...")
    found_metric = None

    for metric in run.metrics():
        if metric.name == METRIC_NAME:
            try:
                # 安全获取 context
                ctx = metric.context.to_dict() if hasattr(metric.context, 'to_dict') else dict(metric.context)
            except:
                ctx = {}

            if ctx.get('subset') == SUBSET_LABEL:
                found_metric = metric
                break

    if not found_metric:
        print(f"❌ Metric not found! Please check configuration.")
        return

    # 4. 提取数据
    steps, values = found_metric.values.sparse_list()

    # 获取 Epochs (关键)
    try:
        epochs = found_metric.epochs.values_list()
    except:
        epochs = []

    # 5. 数据清洗与排序 (核心修复逻辑)
    if len(epochs) == len(values):
        # 构造 DataFrame
        df = pd.DataFrame({
            "epoch": epochs,
            "value": values
        })
        # === 强制按 Epoch 排序 ===
        df = df.sort_values("epoch")

        # 提取排序后的数据
        x_data = df["epoch"].values
        y_data = df["value"].values
        x_label = "迭代轮次"
        print("✅ Data sorted by Epoch.")
    else:
        print("⚠️ Epoch data missing. Sorting by existing order (Step).")
        # 即使没有 Epoch，也尝试按 Step 排序以防万一
        df = pd.DataFrame({"step": steps, "value": values})
        # 过滤掉坏的 Step (负数)
        if df["step"].min() < 0:
            print("⚠️ Detected corrupted steps. Using index as X-axis.")
            x_data = np.arange(len(values))
            x_label = "索引"
        else:
            df = df.sort_values("step")
            x_data = df["step"].values
            x_label = "训练步数"
        y_data = df["value"].values

    print(f"📊 Plotting {len(y_data)} points...")

    # 6. 保存排序后的数据为npy文件
    x_save_path = os.path.join(script_dir, X_DATA_FILENAME)
    y_save_path = os.path.join(script_dir, Y_DATA_FILENAME)

    np.save(x_save_path, x_data)
    np.save(y_save_path, y_data)

    print(f"💾 已保存排序后的x数据: {X_DATA_FILENAME} (形状: {x_data.shape})")
    print(f"💾 已保存排序后的y数据: {Y_DATA_FILENAME} (形状: {y_data.shape})")
    # 6. 绘图 (Nature 风格，无平滑，直接连线)
    font_prop = set_nature_style()

    # 创建宋体字体属性对象（用于坐标轴标签和图例）
    try:
        # 尝试加载宋体字体
        chinese_font_prop = font_manager.FontProperties(family='AR PL UMing CN', size=15)
    except:
        # 如果失败，使用系统默认字体
        chinese_font_prop = font_manager.FontProperties(family='sans-serif', size=15)

    # 创建英文字体属性对象（用于刻度标签）
    if font_prop:
        english_font_prop = font_prop
    else:
        english_font_prop = font_manager.FontProperties(family='Times New Roman', size=15)

    fig, ax = plt.subplots(figsize=(7, 5))

    # 配色 (NPG Blue)
    COLOR_VAL = "black"

    # === 关键绘图语句 ===
    # 直接用 plot 画实线，marker='o' 标记数据点位置
    ax.plot(x_data[0:1100,], y_data[0:1100,],
            color=COLOR_VAL,
            linewidth=1.5,  # 线条粗细
            linestyle='-',  # 实线连接
            label="验证集损失")

    ax.set_yscale('log')
    # 标签与刻度
    # 使用宋体显示中文标签
    ax.set_xlabel(x_label, fontproperties=chinese_font_prop)
    ax.set_ylabel("验证集损失", fontproperties=chinese_font_prop)

    # 使用英文字体显示刻度标签
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(english_font_prop)

    # Y轴格式化 (保留4位小数)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

    # 自动调整 X 轴科学计数法
    if x_label == "训练步数" and max(x_data) > 10000:
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # 图例 - 使用宋体
    ax.legend(prop=chinese_font_prop, loc='upper right')

    # 保存
    save_path = os.path.join(script_dir, OUTPUT_FILENAME)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"✨ Successfully saved raw sorted plot to: {save_path}")


if __name__ == "__main__":
    main()