# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from scipy import stats

def load_source_power(file_path, start=0):
    currents = []
    voltages = []

    with open(file_path, "r") as fp:
        for index, line in enumerate(fp.readlines()):
            if index < 3:
                continue

            arr = line.strip().split(";")

            current = list(map(float, arr[5:12]))
            voltage = list(map(float, arr[13:]))

            currents.append(current)
            voltages.append(voltage)

    currents = np.array(currents)[:, 0]
    voltages = np.array(voltages)[:, 0]

    power = currents * voltages

    rate = round(15 / round(max(power), 2), 2)
    power = power * rate

    return power[start:]


def mean_reversion_stats(x):
    x = pd.Series(x).dropna().reset_index(drop=True)

    x_mean = x.mean()
    deviation = x[:-1] - x_mean
    next_change = x.diff().shift(-1)[:-1]

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        deviation,
        next_change
    )

    return {
        "mean": x.mean(),
        "variance": x.var(ddof=1),
        "std": x.std(ddof=1),
        "lag1_autocorr": x.autocorr(lag=1),
        "mean_reversion_beta": slope,
        "mean_reversion_p_value": p_value
    }


raw_files = [
    "../lynsyn_data_test1.txt",
    "../lynsyn_data_test2.txt",
    "../lynsyn_data_test3.txt",
]

test_names = ["Test 1", "Test 2", "Test 3"]

powers = {}

for name, file in zip(test_names, raw_files):
    powers[name] = load_source_power(file)

min_len = min(len(v) for v in powers.values())

power_df = pd.DataFrame({
    name: values[:min_len]
    for name, values in powers.items()
})

# 1. 三次 Test 的相关性
correlation_df = power_df.corr(method="pearson")

# 2. 每次 Test 的方差、均值、均值回归
stats_df = pd.DataFrame({
    name: mean_reversion_stats(power_df[name])
    for name in power_df.columns
}).T

print("=== 3 Tests Power Correlation ===")
display(correlation_df)

print("=== Variance / Mean / Mean Reversion ===")
display(stats_df)

correlation_df.to_csv(
    f"save_figure/{random_seed}/raw_power_correlation.csv"
)

stats_df.to_csv(
    f"save_figure/{random_seed}/raw_power_variance_mean_reversion.csv"
)