# explain/enrichment.py
import pandas as pd
import os
from collections import Counter

def compute_fragment_enrichment(fragments_tsv_path, output_dir, min_count=2):
    """
    統計 top SHAP fragments 的頻率，找出最常出現的 fragment
    Args:
        fragments_tsv_path: top_shap_fragments.tsv
        output_dir: 統計結果存放資料夾
        min_count: 最少出現次數才列入結果
    Returns:
        enrichment_df: DataFrame with columns ['FragmentSMILES', 'Count', 'Frequency']
    """
    df = pd.read_csv(fragments_tsv_path, sep='\t')
    frag_counts = Counter(df['FragmentSMILES'].dropna())

    # 過濾少量 fragment
    frag_counts = {k: v for k, v in frag_counts.items() if v >= min_count}

    enrichment_df = pd.DataFrame(
        list(frag_counts.items()),
        columns=['FragmentSMILES', 'Count']
    )
    enrichment_df['Frequency'] = enrichment_df['Count'] / enrichment_df['Count'].sum()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "fragment_enrichment.tsv")
    enrichment_df.to_csv(out_path, sep='\t', index=False)
    print(f"[✓] Fragment enrichment saved to {out_path} - enrichment.py:31")

    return enrichment_df


def top_n_fragments(enrichment_df, n=10):
    """
    取前 N 個最常出現的 fragment
    """
    return enrichment_df.sort_values('Count', ascending=False).head(n)
