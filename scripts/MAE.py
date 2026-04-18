import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.font_manager as font_manager
import matplotlib as mpl
import matplotlib.ticker as ticker



# 1. 加载数据
x = torch.load("/home/user/bishe/ftd_lstm/saved_x.pth", map_location=torch.device("cpu")).float().numpy()
y = torch.load("/home/user/bishe/ftd_lstm/saved_y.pth", map_location=torch.device("cpu")).float().numpy()
preds = torch.load("/home/user/bishe/ftd_lstm/saved_preds.pth", map_location=torch.device("cpu")).float().numpy()

print("Shape of x:", x.shape)
print("Shape of y:", y.shape)
print("Shape of preds:", preds.shape)

# 2. 加载归一化参数
imput_min_data = np.load("/home/user/bishe/ftd_lstm/input_scaler_min.npy")
imput_max_data = np.load("/home/user/bishe/ftd_lstm/input_scaler_max.npy")

output_min_data = np.load("/home/user/bishe/ftd_lstm/output_scaler_min.npy")
output_max_data = np.load("/home/user/bishe/ftd_lstm/output_scaler_max.npy")

# 3. 反归一化 (还原到真实物理量级)
x = x * (imput_max_data - imput_min_data) + imput_min_data
y = y * (output_max_data - output_min_data) + output_min_data
preds = preds * (output_max_data - output_min_data) + output_min_data

# ==========================================
# [新增部分] 计算 MAE 并打印
# ==========================================
title_names = ["Cy (侧向力)", "Cl (滚转)", "Cm (俯仰)", "Cn (偏航)", "CD (阻力)", "CL (升力)"]
maes = []

print("\n" + "=" * 40)
print(f"{'Variable':<15} | {'MAE':<15}")
print("-" * 40)

for i in range(6):
    # 提取真实值和预测值 (取第0个时间步)
    real_val = y[:, 0, i]
    pred_val = preds[:, 0, i]

    # 计算该变量的 MAE
    mae_val = np.mean(np.abs(real_val - pred_val))
    maes.append(mae_val)

    print(f"{title_names[i]:<15} | {mae_val:.8f}")

print("-" * 40)
print(f"{'Overall Mean':<15} | {np.mean(maes):.8f}")
print("=" * 40 + "\n")
