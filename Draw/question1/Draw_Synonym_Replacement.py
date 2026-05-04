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

# 模拟数据 (请替换为你的真实数据)
y_lexical_data = [
    [0.6465, 0.6429, 0.6581, 0.6531, 0.6450],  # Fast-DetectGPT
    [0.6662, 0.6670, 0.6680, 0.6683, 0.6663],  # Binoculars
    [0.6432, 0.6681, 0.6640, 0.6806, 0.6723],  # RoBERTa
    [0.7166, 0.7449, 0.7816, 0.7778, 0.8044],  # CoCo
    [0.5228, 0.5272, 0.5911, 0.5886, 0.6222],  # RoBERTa-MPU
    [0.1124, 0.0640, 0.0547, 0.0723, 0.0526],  # BISCOPE
    [0.6628, 0.6661, 0.6666, 0.6638, 0.6665],  # Lastde
    [0.4941, 0.4938, 0.4936, 0.4921, 0.4921],  # SCRN
    [0.7729, 0.7613, 0.7794, 0.7693, 0.7836],  # PRDetect
]
"""
Domain-Specific-Model-Specific_F1
[0.4864, 0.5272, 0.4271, 0.4577, 0.4693],  # Fast-DetectGPT
[0.7864, 0.7918, 0.7788, 0.7984, 0.7718],  # Binoculars
[0.8039, 0.7166, 0.7301, 0.7230, 0.7125],  # RoBERTa
[0.7829, 0.7347, 0.7224, 0.7061, 0.7259],  # CoCo
[0.9068, 0.9066, 0.9093, 0.9058, 0.9018],  # RoBERTa-MPU
[0.7892, 0.6945, 0.7020, 0.6852, 0.6733],  # BISCOPE
[0.4907, 0.5035, 0.6723, 0.6662, 0.6675],  # Lastde
[0.9656, 0.8698, 0.8657, 0.8716, 0.8651],  # SCRN
[0.6977, 0.6893, 0.6777, 0.6740, 0.6523],  # PRDetect
"""

"""
Domain-Specific-Cross-Models_AUROC
[
    [0.5985, 0.5984, 0.6018, 0.6036, 0.6052],  # Fast-DetectGPT
    [0.3993, 0.3968, 0.3990, 0.3962, 0.3971],  # Binoculars
    [0.7683, 0.7650, 0.7746, 0.7295, 0.7669],  # RoBERTa
    [0.9902, 0.9754, 0.9802, 0.9829, 0.9841],  # CoCo
    [0.9890, 0.9879, 0.9854, 0.9875, 0.9854],  # RoBERTa-MPU
    [0.9015, 0.8574, 0.8374, 0.8192, 0.7995],  # BISCOPE
    [0.4020, 0.3901, 0.3899, 0.3876, 0.3858],  # Lastde
    [0.5011, 0.5022, 0.5064, 0.5009, 0.5009],  # SCRN
    [0.9704, 0.9697, 0.9691, 0.9698, 0.9692],  # PRDetect
]
"""

"""
Domain-Specific-Cross-Models_F1
[
    [0.7214, 0.7142, 0.7139, 0.7151, 0.7194],  # Fast-DetectGPT
    [0.6851, 0.6864, 0.6893, 0.6896, 0.6916],  # Binoculars
    [0.7470, 0.6479, 0.4632, 0.6971, 0.4930],  # RoBERTa
    [0.8993, 0.7477, 0.8197, 0.8340, 0.8439],  # CoCo
    [0.8767, 0.8875, 0.8556, 0.8980, 0.8641],  # RoBERTa-MPU
    [0.2052, 0.1954, 0.1931, 0.1915, 0.1879],  # BISCOPE
    [0.6774, 0.6678, 0.6639, 0.6673, 0.6680],  # Lastde
    [0.4924, 0.4923, 0.4918, 0.4910, 0.4899],  # SCRN
    [0.9109, 0.9145, 0.9156, 0.9126, 0.9157],  # PRDetect
]
"""

"""
Cross-Domains-Model-Specific_AUROC
[
    [0.2556, 0.2654, 0.3311, 0.2728, 0.2753],  # Fast-DetectGPT
    [0.2526, 0.2556, 0.2565, 0.2543, 0.2530],  # Binoculars
    [0.8511, 0.8411, 0.8495, 0.8364, 0.8412],  # RoBERTa
    [0.9883, 0.9879, 0.9865, 0.9860, 0.9851],  # CoCo
    [0.9799, 0.9840, 0.9817, 0.9814, 0.9854],  # RoBERTa-MPU
    [0.9545, 0.9568, 0.9461, 0.9369, 0.9286],  # BISCOPE
    [0.1625, 0.1698, 0.1803, 0.1863, 0.1918],  # Lastde
    [0.5076, 0.5122, 0.4987, 0.4960, 0.4952],  # SCRN
    [0.9635, 0.9631, 0.9624, 0.9630, 0.9620],  # PRDetect
]
"""

"""
Cross-Domains-Model-Specific_F1
[
    [0.1821, 0.1791, 0.1774, 0.1787, 0.1780],  # Fast-DetectGPT
    [0.3611, 0.3652, 0.3622, 0.3636, 0.5289],  # Binoculars
    [0.7730, 0.6255, 0.6227, 0.5923, 0.5509],  # RoBERTa
    [0.8819, 0.9221, 0.9261, 0.9357, 0.9347],  # CoCo
    [0.8921, 0.9009, 0.8780, 0.9131, 0.8908],  # RoBERTa-MPU
    [0.9107, 0.8910, 0.8643, 0.8392, 0.8098],  # BISCOPE
    [0.6735, 0.6720, 0.6723, 0.6672, 0.6675],  # Lastde
    [0.6667, 0.6252, 0.6250, 0.6246, 0.6239],  # SCRN
    [0.9073, 0.9060, 0.9080, 0.9088, 0.9093],  # PRDetect
]
"""

"""
Cross-Domains-Cross-Models_AUROC
[
    [0.5029, 0.5023, 0.5038, 0.5051, 0.5053],  # Fast-DetectGPT
    [0.3853, 0.3874, 0.3859, 0.3835, 0.3822],  # Binoculars
    [0.5900, 0.6314, 0.6213, 0.5893, 0.5842],  # RoBERTa
    [0.9348, 0.9409, 0.9533, 0.9568, 0.9567],  # CoCo
    [0.9025, 0.9064, 0.9171, 0.9152, 0.9173],  # RoBERTa-MPU
    [0.8234, 0.7905, 0.7788, 0.7675, 0.7436],  # BISCOPE
    [0.3532, 0.3587, 0.3510, 0.3556, 0.3482],  # Lastde
    [0.4915, 0.5005, 0.4919, 0.4849, 0.5110],  # SCRN
    [0.9067, 0.8974, 0.9056, 0.9032, 0.9010],  # PRDetect
]
"""

"""
Cross-Domains-Cross-Models_F1
[
    [0.6465, 0.6429, 0.6581, 0.6531, 0.6450],  # Fast-DetectGPT
    [0.6662, 0.6670, 0.6680, 0.6683, 0.6663],  # Binoculars
    [0.6432, 0.6681, 0.6640, 0.6806, 0.6723],  # RoBERTa
    [0.7166, 0.7449, 0.7816, 0.7778, 0.8044],  # CoCo
    [0.5228, 0.5272, 0.5911, 0.5886, 0.6222],  # RoBERTa-MPU
    [0.1124, 0.0640, 0.0547, 0.0723, 0.0526],  # BISCOPE
    [0.6628, 0.6661, 0.6666, 0.6638, 0.6665],  # Lastde
    [0.4941, 0.4938, 0.4936, 0.4921, 0.4921],  # SCRN
    [0.7729, 0.7613, 0.7794, 0.7693, 0.7836],  # PRDetect
]
"""

# --- 2. 将数据转换为 Pandas DataFrame (这是使用 Seaborn 的关键!) ---

# --- 2. 构建 DataFrame ---
lexical_df = pd.DataFrame({
    'Replacement_Ratio': np.tile([0, 5, 10, 15, 20], 9),  # ✅ 包含 0%
    'AUROC': np.concatenate(y_lexical_data),
    'Detector': np.repeat(detectors, 5)  # ✅ 每个重复 5 次
})

# --- 3. 绘图 ---
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

plt.figure(figsize=(8, 6))
ax1 = sns.lineplot(
    data=lexical_df,
    x='Replacement_Ratio',
    y='AUROC',
    hue='Detector',
    marker='o',
    palette='tab10'
)

ax1.set_xlabel('Replacement Ratio (%)')
ax1.set_ylabel('AUROC')
# ax1.set_title('(a) Synonym Replacement (WordNet)')
ax1.set_ylim(0.2, 1.0)
ax1.set_xticks([0, 5, 10, 15, 20])  # ✅ 显式设置 x 轴刻度
ax1.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.25))

plt.tight_layout()
plt.savefig('synonym_replacement_seaborn.png', dpi=300, bbox_inches='tight')
plt.show()