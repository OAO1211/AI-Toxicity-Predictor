import os
import pandas as pd

BASE_DIR = "results/dili_binary_labeled - 複製"
MODELS = ["LogReg", "RF", "XGB"]

all_results = []

for model in MODELS:
    path = os.path.join(BASE_DIR, model, "mean_abs_shap.tsv")
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path} - results_analysis.py:12")
        continue

    df = pd.read_csv(path, sep="\t")

    # ✅ 只保留 ECFP bits
    df = df[df["Feature"].str.startswith("ECFP_")].copy()

    # 重新命名欄位成統一格式
    shap_col = [c for c in df.columns if c.startswith("MeanAbsSHAP")][0]
    df = df.rename(columns={shap_col: "MeanAbsSHAP"})
    df["model"] = model

    all_results.append(df)

# 合併三個模型
mean_shap_all = pd.concat(all_results, ignore_index=True)

# 依 SHAP 排序
mean_shap_all = mean_shap_all.sort_values(
    by="MeanAbsSHAP", ascending=False
)

# 存檔
out_path = os.path.join(BASE_DIR, "mean_abs_shap_all_models.tsv")
mean_shap_all.to_csv(out_path, sep="\t", index=False)

print(f"[✓] Saved aggregated SHAP to: {out_path} - results_analysis.py:39")
print(mean_shap_all.head(10))
# Excel 存檔
excel_out = os.path.join(BASE_DIR, "mean_abs_shap_all_models.xlsx")
mean_shap_all.to_excel(excel_out, index=False)
print(f"[✓] Saved aggregated SHAP to Excel: {excel_out} - results_analysis.py:44")
import os
import pandas as pd

BASE_DIR = "results/dili_binary_labeled - 複製"
MODELS = ["LogReg", "RF", "XGB"]
TOP_N = 20  # top N SHAP bits

# 1️⃣ 讀取 fragment 對應表
fragments_dict = {}
for model in MODELS:
    frag_path = os.path.join(BASE_DIR, "fragments", model, "top_shap_fragments.tsv")
    if os.path.exists(frag_path):
        df_frag = pd.read_csv(frag_path, sep="\t")
        # 假設 top_shap_fragments.tsv 裡有 'Bit' 和 'Fragment' 欄
        fragments_dict[model] = df_frag.set_index('BitIndex')['FragmentSMILES'].to_dict()
    else:
        fragments_dict[model] = {}

# 2️⃣ 讀取 mean_abs_shap_all_models.tsv
mean_shap_path = os.path.join(BASE_DIR, "mean_abs_shap_all_models.tsv")
mean_shap_all = pd.read_csv(mean_shap_path, sep="\t")

# 3️⃣ 篩選 top bits per model
top_shap_df_list = []

for model in MODELS:
    df_model = mean_shap_all[mean_shap_all["model"]==model].copy()
    df_model_top = df_model.nlargest(TOP_N, "MeanAbsSHAP")
    df_model_top["Fragment"] = df_model_top["Feature"].apply(lambda x: fragments_dict[model].get(int(x.split("_")[1]), "N/A"))
    df_model_top = df_model_top.rename(columns={"MeanAbsSHAP": f"MeanAbsSHAP_{model}"})
    top_shap_df_list.append(df_model_top[["Feature", "Fragment", f"MeanAbsSHAP_{model}"]])

# 4️⃣ 合併三個模型的 Top SHAP bits
df_merged = top_shap_df_list[0]
for df_next in top_shap_df_list[1:]:
    df_merged = pd.merge(df_merged, df_next, on=["Feature", "Fragment"], how="outer")

# 5️⃣ 存成 Excel
out_excel = os.path.join(BASE_DIR, "Top_SHAP_fragments_all_models.xlsx")
df_merged.to_excel(out_excel, index=False)

print(f"[✓] Saved Top SHAP fragments table to: {out_excel} - results_analysis.py:86")
print(df_merged.head(10))
