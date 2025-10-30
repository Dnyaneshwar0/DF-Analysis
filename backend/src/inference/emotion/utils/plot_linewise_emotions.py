#!/usr/bin/env python3
# (same plotting script as provided earlier, but will use label names from the CSV)
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def make_timeline(df, out_path):
    eps = max(0.1, df['duration'].replace(0, np.nan).median() if not df['duration'].replace(0, np.nan).isna().all() else 0.1)
    starts = df['start'].astype(float).to_numpy()
    durations = df['duration'].astype(float).to_numpy()
    durations = np.where(durations<=0, eps, durations)
    labels = df['dominant'].astype(str).to_list()
    unique_labels = sorted(list(dict.fromkeys(labels)), key=lambda x: x)
    colors = plt.cm.tab20(np.linspace(0,1,max(1,len(unique_labels))))
    fig, ax = plt.subplots(figsize=(12, 3))
    for s,d,lab,text in zip(starts, durations, labels, df['text']):
        idx = unique_labels.index(lab)
        ax.barh(0, d, left=s, height=0.6, color=colors[idx], edgecolor='k', linewidth=0.3)
        txt = f"{lab}: {str(text)[:40].strip()}"
        ax.text(s + d/2, 0, txt, va='center', ha='center', fontsize=8, color='black', clip_on=True)
    ax.set_ylim(-1,1); ax.set_yticks([]); ax.set_xlabel("Time (s)")
    ax.set_title("Caption segments — dominant label (label: text preview)")
    ax.grid(axis='x', linestyle=':', linewidth=0.5)
    legend_patches = [plt.matplotlib.patches.Patch(color=colors[i], label=f"{lab}") for i,lab in enumerate(unique_labels)]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left', title="Label")
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)

def make_stacked(df, out_path, topk=3):
    starts = df['start'].astype(float).to_numpy()
    ends = df['end'].astype(float).to_numpy()
    mid = (starts + ends) / 2.0
    times = mid
    label_probs = {}
    for k in range(1, topk+1):
        lab_col = f"top{k}"
        prob_col = f"top{k}_prob"
        if lab_col in df.columns and prob_col in df.columns:
            for t,lab,prob in zip(times, df[lab_col].astype(str), df[prob_col].astype(float)):
                label_probs.setdefault(lab, []).append((t, float(prob)))
    sorted_idx = np.argsort(times)
    times_sorted = times[sorted_idx]
    labels = sorted(label_probs.keys())
    prob_matrix = np.zeros((len(labels), len(times_sorted)))
    for j,lab in enumerate(labels):
        pairs = dict((round(t,6), p) for t,p in label_probs[lab])
        for i,t in enumerate(times_sorted):
            prob_matrix[j,i] = pairs.get(round(t,6), 0.0)
    fig, ax = plt.subplots(figsize=(12,4))
    colors = plt.cm.tab20(np.linspace(0,1,len(labels)))
    ax.stackplot(times_sorted, prob_matrix, labels=labels, colors=colors, alpha=0.9)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Probability (top-k)")
    ax.set_title(f"Top-{topk} predicted label probabilities over time")
    ax.set_xlim(times_sorted.min() - 0.5, times_sorted.max() + 0.5)
    ax.set_ylim(0, 1.0)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Label")
    ax.grid(axis='y', linestyle=':', linewidth=0.4)
    fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    for col in ["start","end","duration"]: 
        if col not in df.columns:
            raise SystemExit(f"CSV missing column: {col}")
    make_timeline(df, outdir / "linewise_dominant_timeline.png")
    make_stacked(df, outdir / "linewise_top3_stacked.png")
    print("Wrote:", outdir / "linewise_dominant_timeline.png", outdir / "linewise_top3_stacked.png")

if __name__ == "__main__":
    import argparse
    main()
