# 原始数据（按顺序：AUROC, F1, AUROC, F1, ...）
data = [29.32, 49.07, 40.20, 67.74, 16.25, 17.36, 35.32, 66.28]

# 方法1：使用切片（最简洁）
auroc_values = data[0::2]  # 从索引0开始，每隔1个取一个（即第1,3,5,7项）
f1_values   = data[1::2]   # 从索引1开始，每隔1个取一个（即第2,4,6,8项）

avg_auroc = sum(auroc_values) / len(auroc_values)
avg_f1    = sum(f1_values)   / len(f1_values)

print(f"AUROC values: {auroc_values}")
print(f"F1 values:    {f1_values}")
print(f"Average AUROC: {avg_auroc:.2f}")
print(f"Average F1:    {avg_f1:.2f}")