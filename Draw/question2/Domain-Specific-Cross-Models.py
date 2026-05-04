import matplotlib
# 必须在导入 pyplot 之前设置后端
# TkAgg 是最通用的交互式后端
matplotlib.use('TkAgg')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 示例数据结构
data = {
    '5% SR': [7.14, 4.38, 0.32, -0.22],
    '10% SR': [9.88, 8.69, -0.24, 1.85],
    '15% SR': [4.22, 1.39, -0.18, -0.13],
    '20% SR': [5.39, 6.84, -0.70, 3.67],
    'Low PA': [11.63, 4.97, 8.85, 0.85],
    'Medium PA': [27.11, 7.39, 26.00, 1.93],
    'High PA': [26.82, 8.74, 29.03, 3.85]
}
df = pd.DataFrame(data, index=['XSum', 'SQuAD', 'WP', 'SciXGen'])

plt.figure(figsize=(8, 5))
sns.heatmap(df, annot=True, cmap='YlOrRd', fmt=".1f", cbar_kws={'label': 'F1 Score Drop (%)'})
plt.title('Domain-Specific-Cross-Models')
plt.ylabel('Domains')
plt.xlabel('Perturbation Intensity')
plt.show()