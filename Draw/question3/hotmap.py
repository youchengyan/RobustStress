import matplotlib
# 必须在导入 pyplot 之前设置后端
# TkAgg 是最通用的交互式后端
matplotlib.use('TkAgg')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# 1. 准备数据（以 F1 绝对值为例）
# 每一行代表一个实验场景，每一列代表扰动强度
scenarios = ['DSMS', 'DSCM', 'CDMS', 'CDCM']
intensities = ['Clean', '5% SR', '20% SR', 'Low PA', 'Med PA', 'High PA']

# 模拟数据：数值代表 9 个检测器的平均 F1 分数
data = [
    [48.64, 0.88, 0.85, 0.78, 0.65, 0.55], # DSMS: 边界在 High PA 之后
    [0.85, 0.82, 0.75, 0.60, 0.48, 0.35], # DSCM: 边界在 Med PA 附近
    [0.82, 0.78, 0.70, 0.55, 0.45, 0.30], # CDMS: 边界在 Med PA 附近
    [0.75, 0.70, 0.62, 0.45, 0.32, 0.18]  # CDCM: 边界在 Low PA 之后显著提前
]

df = pd.DataFrame(data, index=scenarios, columns=intensities)

# 2. 设置绘图风格
plt.figure(figsize=(10, 6))

# 使用特定的离散色阶：深绿(安全), 橙色(衰减), 深红(失效)
# 设定阈值：F1 > 0.7 (安全), 0.5 < F1 <= 0.7 (衰减), F1 <= 0.5 (失效/边界)
colors = ["#c0392b", "#f39c12", "#27ae60"] # 红, 橙, 绿
cmap = sns.color_palette(colors[::-1], as_cmap=True)

# 3. 绘制热力图
sns.heatmap(df, annot=True, cmap=cmap, cbar_kws={'label': 'Mean F1 Score'},
            linewidths=.5, vmin=0.3, vmax=0.9)

plt.title('Delineating the Failure Boundary of Detectors')
plt.ylabel('Experimental Scenarios')
plt.xlabel('Perturbation Intensity')
plt.show()