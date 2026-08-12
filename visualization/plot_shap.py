# visualization/plot_shap.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# training_shap.tsv / testing_shap.tsv 裡，這幾欄是 metadata，不是特徵
NON_FEATURE_COLS = {"SampleID", "baseline_value", "prediction_prob"}


def plot_mean_abs_shap(mean_shap_path, output_dir="results/plots", top_n=20):
    """
    畫出前 top_n 的 mean |SHAP| 特徵重要性條形圖
    mean_shap_path: mean_abs_shap.tsv
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(mean_shap_path, sep="\t")

    # 假設欄位名稱: Feature / MeanAbsSHAP_MODEL
    feature_col = df.columns[0]
    shap_col = df.columns[1]

    df_sorted = df.sort_values(by=shap_col, ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=shap_col,
        y=feature_col,
        data=df_sorted,
        palette="viridis"
    )
    plt.title(f"Top {top_n} Mean |SHAP| Features")
    plt.xlabel("Mean |SHAP|")
    plt.ylabel("Feature")
    plt.tight_layout()

    out_file = os.path.join(output_dir, f"top{top_n}_mean_abs_shap.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[✓] SHAP bar plot saved to {out_file} - plot_shap.py:41")


def plot_shap_summary(shap_df_path, output_dir="results/plots", top_n=20):
    """
    畫 SHAP summary plot (scatter plot)
    shap_df_path: training_shap.tsv

    注意：特徵欄位用「排除 metadata 欄位」的方式判斷，而不是要求欄位名稱
    以 "ECFP_" 開頭，這樣才能同時支援 ECFP bits 和 descriptors（MolWt、TPSA...）
    這兩種特徵集合。
    """
    import shap  # 確保 shap 已安裝
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    shap_df = pd.read_csv(shap_df_path, sep="\t")

    # 取 feature columns（排除 metadata 欄位，適用 ECFP bits 或 descriptors）
    feature_cols = [c for c in shap_df.columns if c not in NON_FEATURE_COLS]

    # 計算 mean |SHAP|
    mean_abs = shap_df[feature_cols].abs().mean().sort_values(ascending=False)
    top_features = mean_abs.head(top_n).index

    shap_values = shap_df[top_features].values

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        features=shap_df[top_features],
        feature_names=top_features,
        show=False,
        plot_type="dot"
    )

    out_file = os.path.join(output_dir, f"shap_summary_top{top_n}.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] SHAP summary plot saved to {out_file} - plot_shap.py:80")


if __name__ == "__main__":
    # 範例使用
    plot_mean_abs_shap("results/RF/mean_abs_shap.tsv", top_n=20)
    plot_shap_summary("results/RF/fold1/training_shap.tsv", top_n=20)
