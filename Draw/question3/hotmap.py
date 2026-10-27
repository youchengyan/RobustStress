import matplotlib
matplotlib.use('TkAgg')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# 1. prepare data
scenarios = ['DSMS', 'DSCM', 'CDMS', 'CDCM']
intensities = ['Clean', '5% SR', '20% SR', 'Low PA', 'Med PA', 'High PA']

data = [
    [48.64, 0.88, 0.85, 0.78, 0.65, 0.55],
    [0.85, 0.82, 0.75, 0.60, 0.48, 0.35],
    [0.82, 0.78, 0.70, 0.55, 0.45, 0.30],
    [0.75, 0.70, 0.62, 0.45, 0.32, 0.18]
]

df = pd.DataFrame(data, index=scenarios, columns=intensities)

# 2. setting style
plt.figure(figsize=(10, 6))

colors = ["#c0392b", "#f39c12", "#27ae60"]
cmap = sns.color_palette(colors[::-1], as_cmap=True)

# 3. draw picture
sns.heatmap(df, annot=True, cmap=cmap, cbar_kws={'label': 'Mean F1 Score'},
            linewidths=.5, vmin=0.3, vmax=0.9)

plt.title('Delineating the Failure Boundary of Detectors')
plt.ylabel('Experimental Scenarios')
plt.xlabel('Perturbation Intensity')
plt.show()