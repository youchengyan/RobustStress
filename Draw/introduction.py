import matplotlib
# 必须在导入 pyplot 之前设置后端
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 设置配色方案 (提取自你的框架图)
# ==========================================
# Clean Data: 对应 (a) 中的中心圆圈颜色
color_clean = '#DDEBF7'  # 淡蓝色/灰色

# SR Attacks: 对应 (b) 中 Synonym Replacement 的箭头颜色
color_sr_5 = '#A9D08E'   # 浅绿色 (5%)
color_sr_20 = '#548235'  # 深绿色 (20%)

# PA Attacks: 对应 (b) 中 Paraphrasing Attack 的箭头颜色
color_pa_low = '#F8CBAD'   # 浅橙色 (Low)
color_pa_high = '#C65911'  # 深橙色 (High)

# 边框颜色 (用于模拟图中的深绿色边框)
edge_color = '#000'

# ==========================================
# 2. 准备数据 (示例数据，请替换为真实值)
# ==========================================
models = ['Binoculars', 'CoCo', 'PRDetect']
x = np.arange(len(models))  # 模型的位置
width = 0.15  # 柱子的宽度

# 模拟的 ACC 数据 (0-100)
# Clean Data
acc_clean = [78.64, 78.29, 69.77]
# Synonym Replacement (SR)
acc_sr_5 = [77.88, 72.24, 67.77]
acc_sr_20 = [77.18, 72.59, 65.23]
# Paraphrasing Attack (PA)
acc_pa_low = [72.56, 75.59, 50.01]
acc_pa_high = [55.89, 59.71, 25.38]

# ==========================================
# 3. 绘图
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制分组柱状图
# 注意：为了让柱子居中，我们需要减去总宽度的一半 (width * 2.5)
rects1 = ax.bar(x - width*2, acc_clean, width, label='Clean Data', color=color_clean, edgecolor=edge_color, linewidth=1)
rects2 = ax.bar(x - width, acc_sr_5, width, label='10% SR', color=color_sr_5, edgecolor=edge_color, linewidth=1)
rects3 = ax.bar(x, acc_sr_20, width, label='20% SR', color=color_sr_20, edgecolor=edge_color, linewidth=1)
rects4 = ax.bar(x + width, acc_pa_low, width, label='Low PA', color=color_pa_low, edgecolor=edge_color, linewidth=1)
rects5 = ax.bar(x + width*2, acc_pa_high, width, label='High PA', color=color_pa_high, edgecolor=edge_color, linewidth=1)

# ==========================================
# 4. 美化细节
# ==========================================
# 设置标题和标签
ax.set_ylabel('F1', fontsize=24)
# ax.set_title('Robustness Evaluation under Perturbation Attacks', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=20)
# 2. 设置 Y轴刻度标签大小 (添加这一行)
ax.tick_params(axis='y', labelsize=16)
ax.set_ylim(20, 80)  # 设置Y轴范围，留出图例空间

# 添加图例 (放在上方，横向排列，类似框架图的布局)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=5, frameon=True, edgecolor=edge_color, fontsize=15)

# 添加网格线 (仅Y轴，虚线，增加可读性)
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# 去掉顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(edge_color)
ax.spines['bottom'].set_color(edge_color)

plt.tight_layout()

plt.savefig('robustness_results.pdf', format='pdf', bbox_inches='tight')

plt.show()