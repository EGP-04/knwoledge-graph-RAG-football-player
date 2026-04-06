# pip install pandas matplotlib

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Data — paths to your CSV files
# ---------------------------------------------------------------------------

RESPONSES_CSV = "evaluation/responses.csv"
QA_CSV        = "evaluation/qa_pair.csv"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    responses = pd.read_csv(RESPONSES_CSV)
    qa        = pd.read_csv(QA_CSV)

    responses.columns = responses.columns.str.strip()
    qa.columns        = qa.columns.str.strip()

    return responses, qa


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    import re
    if not text or str(text).strip().lower() in ("none", "nan", ""):
        return []
    return re.findall(r"[a-záéíóúàèìòùäëïöüñç']+", str(text).lower())


# ---------------------------------------------------------------------------
# Top-K Match Score (K = 10)
#
# - Uses set overlap
# - Caps matches at 10
# - Uses dynamic denominator:
#       denom = min(10, len(GT tokens))
# ---------------------------------------------------------------------------

def f1_score(predicted: str, ground_truth: str) -> dict:
    pred_tokens = set(tokenize(predicted))
    gt_tokens   = set(tokenize(ground_truth))

    if not gt_tokens:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0}

    correct = len(pred_tokens & gt_tokens)

    # 🔥 Dynamic denominator
    denom = min(10, len(gt_tokens))
    
    # Cap matches at the same threshold to prevent score > 1.0
    correct = min(denom, correct)

    score = correct / denom if denom > 0 else 0.0

    return {
        "f1":        round(score, 3),
        "precision": round(score, 3),
        "recall":    round(score, 3),
    }


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

def run_evaluation(responses: pd.DataFrame, qa: pd.DataFrame) -> pd.DataFrame:
    qa_map = dict(zip(qa["id"], qa["answer"]))
    rows   = []

    for _, row in responses.iterrows():
        gt = str(qa_map.get(row["id"], ""))

        kg_score = f1_score(str(row["KG_RAG_response"]), gt)
        tr_score = f1_score(str(row["trad_RAG_response"]), gt)

        rows.append({
            "id":            row["id"],
            "question":      row["question"],
            "ground_truth":  gt,

            "kg_f1":         kg_score["f1"],
            "kg_precision":  kg_score["precision"],
            "kg_recall":     kg_score["recall"],
            "kg_time":       row["KG_RAG_time"],

            "trad_f1":       tr_score["f1"],
            "trad_precision":tr_score["precision"],
            "trad_recall":   tr_score["recall"],
            "trad_time":     row["trad_RAG_time"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot(eval_df: pd.DataFrame):
    q_ids    = eval_df["id"].tolist()
    q_labels = [f"Q{i}" for i in q_ids]

    kg_times = eval_df["kg_time"].tolist()
    tr_times = eval_df["trad_time"].tolist()
    kg_f1    = eval_df["kg_f1"].tolist()
    tr_f1    = eval_df["trad_f1"].tolist()

    KG_COLOR   = "#378ADD"
    TRAD_COLOR = "#1D9E75"

    # ------------------------------------------------------------------
    # Plot 1: Time comparison
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(q_labels, kg_times, marker="o", linewidth=2, label="KG-RAG")
    plt.plot(q_labels, tr_times, marker="s", linewidth=2, label="Trad-RAG")

    plt.title("Response Time per Question")
    plt.xlabel("Question")
    plt.ylabel("Time (s)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("time_comparison.png", dpi=150)
    plt.show()

    # ------------------------------------------------------------------
    # Plot 2: KG-RAG Score
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(q_labels, kg_f1, marker="o", linewidth=2, color=KG_COLOR)

    for i, val in enumerate(kg_f1):
        plt.annotate(f"{val:.2f}", (q_labels[i], val),
                     textcoords="offset points", xytext=(0, 8),
                     ha='center', fontsize=8)

    plt.axhline(sum(kg_f1)/len(kg_f1), linestyle="--")
    plt.title("KG-RAG — Top-K Token Match Score")
    plt.xlabel("Question")
    plt.ylabel("Score (0–1)")
    plt.ylim(0, 1.1)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("kg_score_plot.png", dpi=150)
    plt.show()

    # ------------------------------------------------------------------
    # Plot 3: Trad-RAG Score
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(q_labels, tr_f1, marker="s", linewidth=2, color=TRAD_COLOR)

    for i, val in enumerate(tr_f1):
        plt.annotate(f"{val:.2f}", (q_labels[i], val),
                     textcoords="offset points", xytext=(0, 8),
                     ha='center', fontsize=8)

    plt.axhline(sum(tr_f1)/len(tr_f1), linestyle="--")
    plt.title("Traditional RAG — Top-K Token Match Score")
    plt.xlabel("Question")
    plt.ylabel("Score (0–1)")
    plt.ylim(0, 1.1)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("trad_score_plot.png", dpi=150)
    plt.show()

    # ------------------------------------------------------------------
    # Plot 4: Average metrics
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    metrics = ["F1", "Precision", "Recall"]

    kg_vals = [
        eval_df["kg_f1"].mean(),
        eval_df["kg_precision"].mean(),
        eval_df["kg_recall"].mean()
    ]

    tr_vals = [
        eval_df["trad_f1"].mean(),
        eval_df["trad_precision"].mean(),
        eval_df["trad_recall"].mean()
    ]

    x = range(len(metrics))
    width = 0.35

    plt.bar([i - width/2 for i in x], kg_vals, width=width, label="KG-RAG")
    plt.bar([i + width/2 for i in x], tr_vals, width=width, label="Trad-RAG")

    plt.xticks(x, metrics)
    plt.ylabel("Average Score")
    plt.title("Average Metrics Comparison")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("avg_metrics.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------

def print_summary(eval_df: pd.DataFrame):
    print("\n" + "="*80)
    print("TOP-K TOKEN MATCH EVALUATION SUMMARY (Dynamic GT)")
    print("="*80)

    print(f"  {'Q':<4} {'KG Score':>10} {'KG Time':>10}  |  {'Tr Score':>10} {'Tr Time':>10}")
    print(f"  {'-'*60}")

    for _, row in eval_df.iterrows():
        print(f"  Q{int(row['id']):<3} {row['kg_f1']:>10.3f} {row['kg_time']:>9.1f}s  |  {row['trad_f1']:>10.3f} {row['trad_time']:>9.1f}s")

    print(f"  {'-'*60}")

    print(f"  {'AVG':<4} {eval_df['kg_f1'].mean():>10.3f} {eval_df['kg_time'].mean():>9.1f}s  |  {eval_df['trad_f1'].mean():>10.3f} {eval_df['trad_time'].mean():>9.1f}s")

    print("="*80)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    responses, qa = load_data()
    eval_df       = run_evaluation(responses, qa)

    print_summary(eval_df)
    plot(eval_df)