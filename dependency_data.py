import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_sensor_file(path):
    df = pd.read_csv(path, sep=";", skiprows=2)
    df = df.dropna(axis=1, how="all")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def remove_pc012(df):
    remove_cols = ["pc 0", "pc 1", "pc 2", "pc 3"]
    return df.drop(columns=remove_cols, errors="ignore")


def align_to_common_time(dfs, time_col="Time"):
    start = max(df[time_col].min() for df in dfs)
    end = min(df[time_col].max() for df in dfs)

    base = dfs[0].sort_values(time_col)
    common_time = base[
        (base[time_col] >= start) &
        (base[time_col] <= end)
    ][time_col]

    aligned = []

    for df in dfs:
        df = df.sort_values(time_col)
        new_df = pd.DataFrame({time_col: common_time})

        for col in df.columns:
            if col == time_col:
                continue

            new_df[col] = np.interp(
                common_time,
                df[time_col],
                df[col]
            )

        aligned.append(new_df)

    return aligned


def overall_correlation(df_a, df_b):
    common_cols = [
        col for col in df_a.columns
        if col != "Time" and col in df_b.columns
    ]

    a = df_a[common_cols].to_numpy().flatten()
    b = df_b[common_cols].to_numpy().flatten()

    mask = ~np.isnan(a) & ~np.isnan(b)

    return np.corrcoef(a[mask], b[mask])[0, 1]


file1 = "lynsyn_data1.txt"
file2 = "lynsyn_data2.txt"
file3 = "lynsyn_data3.txt"

df1 = read_sensor_file(file1)
df2 = read_sensor_file(file2)
df3 = read_sensor_file(file3)

# 去掉 pc 0, pc 1, pc 2, pc 3
df1 = remove_pc012(df1)
df2 = remove_pc012(df2)
df3 = remove_pc012(df3)

# 按 Time 对齐
df1, df2, df3 = align_to_common_time([df1, df2, df3])

corr_12 = overall_correlation(df1, df2)
corr_13 = overall_correlation(df1, df3)
corr_23 = overall_correlation(df2, df3)

print("file1 vs file2:", corr_12)
print("file1 vs file3:", corr_13)
print("file2 vs file3:", corr_23)

# 如果只想要一个总相关性值
final_corr = (corr_12 + corr_13 + corr_23) / 3
print("overall:", final_corr)

# 组成 3x3 相关性矩阵
corr_matrix = pd.DataFrame(
    [
        [1.0,     corr_12, corr_13],
        [corr_12, 1.0,     corr_23],
        [corr_13, corr_23, 1.0]
    ],
    index=["file1", "file2", "file3"],
    columns=["file1", "file2", "file3"]
)

# 转成百分比
corr_percent = corr_matrix * 100

print("\nCorrelation Matrix (%):")
print(corr_percent)

# 画百分比热力图
plt.figure(figsize=(6, 5))
sns.heatmap(
    corr_percent,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-100,
    vmax=100,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Pearson Correlation (%)"}
)

plt.title("Pearson Correlation Heatmap Between Three Files (%)")
plt.tight_layout()
plt.show()