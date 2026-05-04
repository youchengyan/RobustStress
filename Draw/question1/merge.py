import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- 1. 检测器名称 ---
detectors = [
    "Fast-DetectGPT", "Binoculars", "RoBERTa", "CoCo",
    "RoBERTa-MPU", "BISCOPE", "Lastde", "SCRN", "PRDetect"
]

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 9})

# ==============================
# 第一部分：Synonym Replacement 数据 (5 points: 0,5,10,15,20)
# ==============================
x_synonym = [0, 5, 10, 15, 20]

synonym_auroc = [
    [0.5985, 0.5984, 0.6018, 0.6036, 0.6052],
    [0.3993, 0.3968, 0.3990, 0.3962, 0.3971],
    [0.7683, 0.7650, 0.7746, 0.7295, 0.7669],
    [0.9902, 0.9754, 0.9802, 0.9829, 0.9841],
    [0.9890, 0.9879, 0.9854, 0.9875, 0.9854],
    [0.9015, 0.8574, 0.8374, 0.8192, 0.7995],
    [0.4020, 0.3901, 0.3899, 0.3876, 0.3858],
    [0.5011, 0.5022, 0.5064, 0.5009, 0.5009],
    [0.9704, 0.9697, 0.9691, 0.9698, 0.9692]
]

synonym_f1 = [
    [0.7214, 0.7142, 0.7139, 0.7151, 0.7194],
    [0.6851, 0.6864, 0.6893, 0.6896, 0.6916],
    [0.7470, 0.6479, 0.4632, 0.6971, 0.4930],
    [0.8993, 0.7477, 0.8197, 0.8340, 0.8439],
    [0.8767, 0.8875, 0.8556, 0.8980, 0.8641],
    [0.2052, 0.1954, 0.1931, 0.1915, 0.1879],
    [0.6774, 0.6678, 0.6639, 0.6673, 0.6680],
    [0.4924, 0.4923, 0.4918, 0.4910, 0.4899],
    [0.9109, 0.9145, 0.9156, 0.9126, 0.9157]
]

# ==============================
# 第二部分：Paraphrase Attack 数据 (4 levels: Clean, Low, Medium, High)
# ==============================
paraphrase_labels = ['Clean', 'Low', 'Medium', 'High']

paraphrase_data = {
    ('Qwen2.5-7B', 'AUROC'): [
        [0.5985, 0.5944, 0.5957, 0.6054],
        [0.3993, 0.3911, 0.3973, 0.4238],
        [0.7683, 0.7673, 0.7804, 0.7673],
        [0.9902, 0.9535, 0.9106, 0.8861],
        [0.9890, 0.9618, 0.9331, 0.9156],
        [0.9015, 0.7656, 0.7735, 0.7672],
        [0.4020, 0.3903, 0.4186, 0.4433],
        [0.5011, 0.5115, 0.4989, 0.5023],
        [0.9704, 0.9159, 0.8731, 0.8458]
    ],
    ('Qwen2.5-7B', 'F1'): [
        [0.7214, 0.7124, 0.6903, 0.7014],
        [0.6851, 0.6847, 0.6780, 0.6868],
        [0.7470, 0.7284, 0.7083, 0.7091],
        [0.8993, 0.6805, 0.4714, 0.3543],
        [0.8767, 0.7208, 0.5524, 0.4961],
        [0.2052, 0.1296, 0.1226, 0.1290],
        [0.6774, 0.6681, 0.6601, 0.6570],
        [0.4924, 0.4712, 0.4779, 0.4724],
        [0.9109, 0.8310, 0.7537, 0.6938]
    ],
    ('Llama3-8B', 'AUROC'): [
        [0.5985, 0.5273, 0.5109, 0.5264],
        [0.3993, 0.3706, 0.3597, 0.3778],
        [0.7683, 0.8073, 0.7790, 0.7551],
        [0.9902, 0.9544, 0.9069, 0.8584],
        [0.9890, 0.9593, 0.9196, 0.8798],
        [0.9015, 0.7604, 0.7005, 0.6702],
        [0.4020, 0.3435, 0.3636, 0.4038],
        [0.5011, 0.5130, 0.5055, 0.4948],
        [0.9704, 0.9070, 0.8227, 0.7792]
    ],
    ('Llama3-8B', 'F1'): [
        [0.7214, 0.7407, 0.7007, 0.6511],
        [0.6851, 0.7210, 0.6999, 0.6707],
        [0.7470, 0.7101, 0.6335, 0.5807],
        [0.8993, 0.7210, 0.5738, 0.4472],
        [0.8767, 0.7125, 0.5597, 0.4763],
        [0.2052, 0.0982, 0.0741, 0.0691],
        [0.6774, 0.6586, 0.6639, 0.6571],
        [0.4924, 0.4284, 0.4504, 0.4847],
        [0.9109, 0.8326, 0.7175, 0.6386]
    ],
    ('Qwen3-14B', 'AUROC'): [
        [0.5985, 0.5900, 0.6191, 0.6315],
        [0.3993, 0.3915, 0.4068, 0.4258],
        [0.7683, 0.7789, 0.8004, 0.7815],
        [0.9902, 0.9785, 0.9256, 0.9029],
        [0.9890, 0.9829, 0.9535, 0.9361],
        [0.9015, 0.8919, 0.8460, 0.8216],
        [0.4020, 0.3834, 0.4394, 0.4699],
        [0.5011, 0.5641, 0.4997, 0.6360],
        [0.9704, 0.9645, 0.9202, 0.8961]
    ],
    ('Qwen3-14B', 'F1'): [
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
}

# ==============================
# 辅助函数：创建 DataFrame
# ==============================
def create_synonym_df(data, metric):
    return pd.DataFrame({
        'Replacement_Ratio': np.tile(x_synonym, len(detectors)),
        metric: np.concatenate(data),
        'Detector': np.repeat(detectors, len(x_synonym))
    })

def create_paraphrase_df(data, metric):
    return pd.DataFrame({
        'Attack_Level': np.tile(paraphrase_labels, len(detectors)),
        metric: np.concatenate(data),
        'Detector': np.repeat(detectors, len(paraphrase_labels))
    })

# ==============================
# 构建所有 DataFrame
# ==============================
df_syn_auroc = create_synonym_df(synonym_auroc, 'AUROC')
df_syn_f1    = create_synonym_df(synonym_f1, 'F1')

# 设置分类顺序
df_syn_auroc['Replacement_Ratio'] = pd.Categorical(df_syn_auroc['Replacement_Ratio'], categories=x_synonym, ordered=True)
df_syn_f1['Replacement_Ratio']    = pd.Categorical(df_syn_f1['Replacement_Ratio'], categories=x_synonym, ordered=True)

paraphrase_dfs = {}
for key, data in paraphrase_data.items():
    metric = key[1]
    df = create_paraphrase_df(data, metric)
    df['Attack_Level'] = pd.Categorical(df['Attack_Level'], categories=paraphrase_labels, ordered=True)
    paraphrase_dfs[key] = df

# ==============================
# 绘图：2 行 × 4 列
# ==============================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for ax in axes.flat:  # axes.flat 将 2x4 数组展平为一维，方便遍历
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)  # 可选：加粗边框（默认太细）

axes = axes.flatten()

# 第1行：Synonym AUROC, Synonym F1, Qwen2.5-7B AUROC, Qwen2.5-7B F1
plot_configs = [
    (df_syn_auroc, 'Replacement_Ratio', 'AUROC', '(a) Synonym Replacement (AUROC)', x_synonym),
    (df_syn_f1,    'Replacement_Ratio', 'F1',    '(b) Synonym Replacement (F1)',    x_synonym),
    (paraphrase_dfs[('Qwen2.5-7B', 'AUROC')], 'Attack_Level', 'AUROC', '(c) Paraphrase Attack by Qwen2.5-7B (AUROC)', paraphrase_labels),
    (paraphrase_dfs[('Qwen2.5-7B', 'F1')],    'Attack_Level', 'F1',    '(d) Paraphrase Attack by Qwen2.5-7B (F1)',    paraphrase_labels),
    (paraphrase_dfs[('Llama3-8B', 'AUROC')], 'Attack_Level', 'AUROC', '(e) Paraphrase Attack by Llama3-8B (AUROC)', paraphrase_labels),
    (paraphrase_dfs[('Llama3-8B', 'F1')],    'Attack_Level', 'F1',    '(f) Paraphrase Attack by Llama3-8B (F1)',    paraphrase_labels),
    (paraphrase_dfs[('Qwen3-14B', 'AUROC')], 'Attack_Level', 'AUROC', '(g) Paraphrase Attack by Qwen3-14B (AUROC)', paraphrase_labels),
    (paraphrase_dfs[('Qwen3-14B', 'F1')],    'Attack_Level', 'F1',    '(h) Paraphrase Attack by Qwen3-14B (F1)',    paraphrase_labels)
]

for idx, (df, x_col, y_col, title, x_ticks) in enumerate(plot_configs):
    ax = axes[idx]
    sns.lineplot(
        data=df,
        x=x_col,
        y=y_col,
        hue='Detector',
        marker='o',
        palette='tab10',
        ax=ax,
        legend=False  # 统一图例
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('')
    ax.set_ylabel(y_col)
    ax.set_xticks(x_ticks)
    if 'Replacement' in title:
        ax.set_ylim(0.17, 1.0)
    else:
        ax.set_ylim(0.05, 1.0)

# ==============================
# 添加统一图例
# ==============================
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.03))

from matplotlib.lines import Line2D

# 使用 tab10 前9种颜色
colors = plt.cm.tab10.colors[:9]
handles = [
    Line2D(
        [0], [0],                     # 虚拟数据点
        color=c,                      # 线条颜色
        marker='o',                   # 圆形标记
        markersize=5,                 # 点大小
        markerfacecolor=c,            # 实心填充色 = 线色
        # markeredgecolor='black',      # 可选：加黑边更清晰
        markeredgewidth=0.5,
        linewidth=1.5,                # 线宽
        label=det
    )
    for c, det in zip(colors, detectors)
]
labels = detectors

fig.legend(
    handles, labels,
    loc='lower center',
    ncol=9,                      # 一行9个（或改为 ncol=5 分两行）
    bbox_to_anchor=(0.5, 0.045), # 位置更低
    frameon=True,               # 无边框 → 匹配你的截图
    fancybox=False,
    shadow=False,
    fontsize=9,
    handlelength=2,
    handletextpad=0.3
)

# ==============================
# 调整布局并保存
# ==============================
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('combined_8_subplots.png', dpi=300, bbox_inches='tight')
print("✅ 8 张子图已保存为 combined_8_subplots.png")

plt.show()