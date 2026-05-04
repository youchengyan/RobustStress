import matplotlib
# 必须在导入 pyplot 之前设置后端
matplotlib.use('TkAgg')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# 1. 准备数据 (将你提供的三个数据集合整理好)
# ==========================================

# 数据集 1
data1 = {
    '5% SR': [3.53, -0.62, 2.62, 4.13],
    '10% SR': [1.86, 0.77, 2.1, 8.19],
    '15% SR': [2.89, 1.05, -0.79, 9.66],
    '20% SR': [2.44, 2.28, 0.52, 9.73],
    'Low PA': [5.64, 5.4, 2.75, 2.12],
    'Medium PA': [29.17, 10.82, 3.28, 5.95],
    'High PA': [24.38, 18.79, 19.26, 7.76]
}
df1 = pd.DataFrame(data1, index=['XSum', 'SQuAD', 'WP', 'SciXGen'])

# 数据集 2
data2 = {
    '5% SR': [7.14, 4.38, 0.32, -0.22],
    '10% SR': [9.88, 8.69, -0.24, 1.85],
    '15% SR': [4.22, 1.39, -0.18, -0.13],
    '20% SR': [5.39, 6.84, -0.70, 3.67],
    'Low PA': [11.63, 4.97, 8.85, 0.85],
    'Medium PA': [27.11, 7.39, 26.00, 1.93],
    'High PA': [26.82, 8.74, 29.03, 3.85]
}
df2 = pd.DataFrame(data2, index=['XSum', 'SQuAD', 'WP', 'SciXGen'])

# 数据集 3
data3 = {
    '5% SR': [0.82, 1.33, -0.38, 5.6],
    '10% SR': [2.62, -0.5, -0.45, 7.96],
    '15% SR': [1.1, 1.19, -1.05, 8.92],
    '20% SR': [-5.47, 0.58, 1.92, 9.52],
    'Low PA': [-0.44, 5.1, 4.5, 6.15],
    'Medium PA': [13.6, 10.26, -0.1, 16.67],
    'High PA': [12.76, 12.67, 2.39, 20.75]
}
df3 = pd.DataFrame(data3, index=['GPT-3.5 Turbo', 'GLM-130B', 'LLaMA-13B', 'GPT-J-6B'])

# ==========================================
# 2. 创建画布与绘图 (一行三列)
# ==========================================

# 创建画布，1行3列，调整整体大小 (宽24英寸，高5英寸)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- 绘制第一张图 ---
sns.heatmap(df1, ax=axes[0], annot=True, cmap='YlOrRd', fmt=".1f", cbar_kws={'label': 'F1 Score Drop (%)'})
axes[0].set_title('Domain-Specific-Model-Specific')
axes[0].set_ylabel('Domains')
axes[0].set_xlabel('Perturbation Intensity')

# --- 绘制第二张图 ---
sns.heatmap(df2, ax=axes[1], annot=True, cmap='YlOrRd', fmt=".1f", cbar_kws={'label': 'F1 Score Drop (%)'})
axes[1].set_title('Domain-Specific-Cross-Models')
axes[1].set_ylabel('Domains')
axes[1].set_xlabel('Perturbation Intensity')

# --- 绘制第三张图 ---
sns.heatmap(df3, ax=axes[2], annot=True, cmap='YlOrRd', fmt=".1f", cbar_kws={'label': 'F1 Score Drop (%)'})
axes[2].set_title('Cross-Domains-Model-Specific')
axes[2].set_ylabel('Domains')
axes[2].set_xlabel('Perturbation Intensity')

# ==========================================
# 3. 调整布局并显示
# ==========================================
# 自动调整子图参数，防止标签重叠
plt.tight_layout()

# 显示图片
plt.show()

# 如果需要保存图片，取消下面这行的注释
plt.savefig('combined_heatmaps.png', dpi=300)