import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- 1. 准备数据 ---
detectors = [
    "Fast-DetectGPT", "Binoculars", "RoBERTa", "CoCo",
    "RoBERTa-MPU", "BISCOPE", "Lastde", "SCRN", "PRDetect"
]

# ✅ 替换为你的 Paraphrase Attack 数据（每个模型 4 个值）
# 格式: [Clean, Low, Medium, High]
y_paraphrase_data = [
    [0.7214, 0.7224, 0.7176, 0.7146],
    [0.6851, 0.6851, 0.6816, 0.6820],
    [0.7470, 0.7464, 0.7227, 0.7052],
    [0.8993, 0.8114, 0.4584, 0.3882],
    [0.8767, 0.8448, 0.5858, 0.5273],
    [0.2052, 0.1944, 0.1635, 0.1496],
    [0.6774, 0.6702, 0.6632, 0.6600],
    [0.4924, 0.4880, 0.4818, 0.5501],
    [0.9109, 0.9011, 0.8257, 0.7696]
]

# Qwen7B
"""
paraphrase_Domain-Specific-Cross-Models_AUROC
[
    [0.5985, 0.5944, 0.5957, 0.6054],
    [0.3993, 0.3911, 0.3973, 0.4238],
    [0.7683, 0.7673, 0.7804, 0.7673],
    [0.9902, 0.9535, 0.9106, 0.8861],
    [0.9890, 0.9618, 0.9331, 0.9156],
    [0.9015, 0.7656, 0.7735, 0.7672],
    [0.4020, 0.3903, 0.4186, 0.4433],
    [0.5011, 0.5115, 0.4989, 0.5023],
    [0.9704, 0.9159, 0.8731, 0.8458]
]
    [0.5985],
    [0.3993],
    [0.7683],
    [0.9902],
    [0.9890],
    [0.9015],
    [0.4020],
    [0.5011],
    [0.9704]
"""

"""
paraphrase_Domain-Specific-Cross-Models_F1
[
    [0.7214, 0.7124, 0.6903, 0.7014],
    [0.6851, 0.6847, 0.6780, 0.6868],
    [0.7470, 0.7284, 0.7083, 0.7091],
    [0.8993, 0.6805, 0.4714, 0.3543],
    [0.8767, 0.7208, 0.5524, 0.4961],
    [0.2052, 0.1296, 0.1226, 0.1290],
    [0.6774, 0.6681, 0.6601, 0.6570],
    [0.4924, 0.4712, 0.4779, 0.4724],
    [0.9109, 0.8310, 0.7537, 0.6938]
]
    [0.7214],
    [0.6851],
    [0.7470],
    [0.8993],
    [0.8767],
    [0.2052],
    [0.6774],
    [0.4924],
    [0.9109]
"""

# llama3 8B
"""
paraphrase_Domain-Specific-Cross-Models_AUROC
[
    [0.5985, 0.5273, 0.5109, 0.5264],
    [0.3993, 0.3706, 0.3597, 0.3778],
    [0.7683, 0.8073, 0.7790, 0.7551],
    [0.9902, 0.9544, 0.9069, 0.8584],
    [0.9890, 0.9593, 0.9196, 0.8798],
    [0.9015, 0.7604, 0.7005, 0.6702],
    [0.4020, 0.3435, 0.3636, 0.4038],
    [0.5011, 0.5130, 0.5055, 0.4948],
    [0.9704, 0.9070, 0.8227, 0.7792]
]
"""
"""
paraphrase_Domain-Specific-Cross-Models_F1
[
    [0.7214, 0.7407, 0.7007, 0.6511],
    [0.6851, 0.7210, 0.6999, 0.6707],
    [0.7470, 0.7101, 0.6335, 0.5807],
    [0.8993, 0.7210, 0.5738, 0.4472],
    [0.8767, 0.7125, 0.5597, 0.4763],
    [0.2052, 0.0982, 0.0741, 0.0691],
    [0.6774, 0.6586, 0.6639, 0.6571],
    [0.4924, 0.4284, 0.4504, 0.4847],
    [0.9109, 0.8326, 0.7175, 0.6386]
]
"""

# Qwen 14B
"""
paraphrase_Domain-Specific-Cross-Models_AUROC
[
    [0.5985, 0.5900, 0.6191, 0.6315],
    [0.3993, 0.3915, 0.4068, 0.4258],
    [0.7683, 0.7789, 0.8004, 0.7815],
    [0.9902, 0.9785, 0.9256, 0.9029],
    [0.9890, 0.9829, 0.9535, 0.9361],
    [0.9015, 0.8919, 0.8460, 0.8216],
    [0.4020, 0.3834, 0.4394, 0.4699],
    [0.5011, 0.5641, 0.4997, 0.6360],
    [0.9704, 0.9645, 0.9202, 0.8961]
]
"""
"""
paraphrase_Domain-Specific-Cross-Models_F1
[
    [0.7214, 0.7224, 0.7176, 0.7146],
    [0.6851, 0.6851, 0.6816, 0.6820],
    [0.7470, 0.7464, 0.7227, 0.7052],
    [0.8993, 0.8114, 0.4584, 0.3882],
    [0.8767, 0.8448, 0.5858, 0.5273],
    [0.2052, 0.1944, 0.1635, 0.1496],
    [0.6774, 0.6702, 0.6632, 0.6600],
    [0.4924, 0.4880, 0.4818, 0.5501],
    [0.9109, 0.9011, 0.8257, 0.7696]
]
"""


# --- 2. 构建 DataFrame ---
x_labels = ['Clean', 'Low', 'Medium', 'High']

paraphrase_df = pd.DataFrame({
    'Attack_Level': np.tile(x_labels, len(detectors)),  # 4 labels × 9 detectors
    'AUROC': np.concatenate(y_paraphrase_data),
    'Detector': np.repeat(detectors, len(x_labels))   # 每个 detector 重复 4 次
})

# ✅ 确保 x 轴按指定顺序排列（避免 seaborn 自动排序）
paraphrase_df['Attack_Level'] = pd.Categorical(
    paraphrase_df['Attack_Level'],
    categories=x_labels,
    ordered=True
)

# --- 3. 绘图 ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

plt.figure(figsize=(8, 6))
ax1 = sns.lineplot(
    data=paraphrase_df,
    x='Attack_Level',
    y='AUROC',
    hue='Detector',
    marker='o',
    palette='tab10'
)

ax1.set_xlabel('Paraphrase Attack Level')
ax1.set_ylabel('AUROC')
# ax1.set_title('(b) Paraphrase Attack')
ax1.set_ylim(0.0, 1.0)

# 图例放在下方
ax1.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.25))

plt.tight_layout()
plt.savefig('paraphrase_attack_seaborn.png', dpi=300, bbox_inches='tight')
plt.show()