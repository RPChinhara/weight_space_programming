ROOT=${ROOT:-$PWD/assets}
M=$ROOT/models; D=$ROOT/data
mkdir -p "$M" "$D"

# ---- Models ----
hf download Qwen/Qwen3.5-0.8B   --local-dir "$M/Qwen3.5-0.8B"     # target (~1.6 GB)
hf download Qwen/Qwen3.5-4B     --local-dir "$M/Qwen3.5-4B"       # editor, small (~8 GB)
hf download Qwen/Qwen3.5-9B     --local-dir "$M/Qwen3.5-9B"       # editor, large (~18 GB)
# hf download Qwen/Qwen2.5-7B-Instruct --local-dir "$M/Qwen2.5-7B-Instruct"  # literal-7B alternative
hf download google/gemma-3-270m --local-dir "$M/gemma-3-270m"     # stretch target (gated)

# ---- Practice data (editor may see) ----
hf download openai/gsm8k                --repo-type dataset --local-dir "$D/gsm8k"         # use main/train
hf download EleutherAI/hendrycks_math   --repo-type dataset --local-dir "$D/math"          # use train splits
hf download AI-MO/aimo-validation-aime  --repo-type dataset --local-dir "$D/aime_hist"     # historical AIME

# ---- Held-out / collateral (editor must never see) ----
hf download HuggingFaceH4/MATH-500      --repo-type dataset --local-dir "$D/math500"
# GSM8K test is in $D/gsm8k (main/test)
hf download google/IFEval               --repo-type dataset --local-dir "$D/ifeval"
hf download Salesforce/wikitext         --repo-type dataset --local-dir "$D/wikitext" \
  --include "wikitext-103-raw-v1/*"                                                        # KL / perplexity sample

# ---- Overlap check: MATH-500 is drawn from the MATH test set, so MATH train should not overlap; this confirms it ----
python - <<'EOF'
import json, glob, pandas as pd, os
D = os.environ.get("ROOT", "assets") + "/data"
def load(p):
    fs = glob.glob(p, recursive=True)
    return pd.concat([pd.read_parquet(f) if f.endswith(".parquet") else pd.read_json(f, lines=True) for f in fs])
m500 = load(f"{D}/math500/**/*.jsonl")
train = load(f"{D}/math/**/train-*.parquet")
overlap = set(m500["problem"]) & set(train["problem"])
print(f"MATH-500: {len(m500)}  MATH train: {len(train)}  overlap: {len(overlap)}")
EOF

du -sh "$M"/* "$D"/*
