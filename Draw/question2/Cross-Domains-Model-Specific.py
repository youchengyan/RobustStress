import matplotlib
# 必须在导入 pyplot 之前设置后端
# TkAgg 是最通用的交互式后端
matplotlib.use('TkAgg')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 示例数据结构
data = {
    '5% SR': [0.82, 1.33, -0.38, 5.6],
    '10% SR': [2.62, -0.5, -0.45, 7.96],
    '15% SR': [1.1, 1.19, -1.05, 8.92],
    '20% SR': [-5.47, 0.58, 1.92, 9.52],
    'Low PA': [-0.44, 5.1, 4.5, 6.15],
    'Medium PA': [13.6, 10.26, -0.1, 16.67],
    'High PA': [12.76, 12.67, 2.39, 20.75]
}
df = pd.DataFrame(data, index=['GPT-3.5 Turbo', 'GLM-130B', 'LLaMA-13B', 'GPT-J-6B'])

plt.figure(figsize=(8, 5))
sns.heatmap(df, annot=True, cmap='YlOrRd', fmt=".1f", cbar_kws={'label': 'F1 Score Drop (%)'})
plt.title('Cross-Domains-Model-Specific')
plt.ylabel('Domains')
plt.xlabel('Perturbation Intensity')
plt.show()