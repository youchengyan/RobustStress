import matplotlib
# 必须在导入 pyplot 之前设置后端
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. setting colors
# ==========================================
color_clean = '#DDEBF7'

color_sr_5 = '#A9D08E'
color_sr_20 = '#548235'

color_pa_low = '#F8CBAD'
color_pa_high = '#C65911'

edge_color = '#000'

# ==========================================
# 2. prepare data
# ==========================================
models = ['Binoculars', 'CoCo', 'PRDetect']
x = np.arange(len(models))
width = 0.15

# Clean Data
acc_clean = [78.64, 78.29, 69.77]
# Synonym Replacement (SR)
acc_sr_5 = [77.88, 72.24, 67.77]
acc_sr_20 = [77.18, 72.59, 65.23]
# Paraphrasing Attack (PA)
acc_pa_low = [72.56, 75.59, 50.01]
acc_pa_high = [55.89, 59.71, 25.38]

# ==========================================
# 3. draw picture
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width*2, acc_clean, width, label='Clean Data', color=color_clean, edgecolor=edge_color, linewidth=1)
rects2 = ax.bar(x - width, acc_sr_5, width, label='10% SR', color=color_sr_5, edgecolor=edge_color, linewidth=1)
rects3 = ax.bar(x, acc_sr_20, width, label='20% SR', color=color_sr_20, edgecolor=edge_color, linewidth=1)
rects4 = ax.bar(x + width, acc_pa_low, width, label='Low PA', color=color_pa_low, edgecolor=edge_color, linewidth=1)
rects5 = ax.bar(x + width*2, acc_pa_high, width, label='High PA', color=color_pa_high, edgecolor=edge_color, linewidth=1)

# ==========================================
# 4. process details
# ==========================================
# title and label
ax.set_ylabel('F1', fontsize=24)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=20)
ax.tick_params(axis='y', labelsize=16)
ax.set_ylim(20, 80)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=5, frameon=True, edgecolor=edge_color, fontsize=15)

ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(edge_color)
ax.spines['bottom'].set_color(edge_color)

plt.tight_layout()

plt.savefig('robustness_results.pdf', format='pdf', bbox_inches='tight')

plt.show()