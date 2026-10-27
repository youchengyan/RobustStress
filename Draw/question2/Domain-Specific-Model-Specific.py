import matplotlib
matplotlib.use('TkAgg')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = {
    '5% SR': [3.53, -0.62, 2.62, 4.13],
    '10% SR': [1.86, 0.77, 2.1, 8.19],
    '15% SR': [2.89, 1.05, -0.79, 9.66],
    '20% SR': [2.44, 2.28, 0.52, 9.73],
    'Low PA': [5.64, 5.4, 2.75, 2.12],
    'Medium PA': [29.17, 10.82, 3.28, 5.95],
    'High PA': [24.38, 18.79, 19.26, 7.76]
}
df = pd.DataFrame(data, index=['XSum', 'SQuAD', 'WP', 'SciXGen'])

plt.figure(figsize=(10, 5))

sns.heatmap(df, annot=True, cmap='YlOrRd', fmt=".1f", cbar_kws={'label': 'F1 Score Drop (%)'})
plt.title('Domain-Specific-Model-Specific')
plt.ylabel('Domains')
plt.xlabel('Perturbation Intensity')
plt.show()