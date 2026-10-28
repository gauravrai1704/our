"""Collect runs/*/results.json into paper tables (mean ± std over seeds).

    python aggregate.py                      # all runs
    python aggregate.py --runs runs --out runs/tables

Runs named <variant>_s<seed> are grouped by variant. Writes <out>.md and <out>.tex:
  Table 1  final per-dataset PSNR / SSIM (oracle routing), ACC, BWT, FGT,
           and the mean drop vs the single-task specialists (if specialist_* runs exist)
  Table 2  task-free inference: ACC with oracle vs inferred vs soft routing, routing accuracy
  Table 3  held-out compound degradations (RQ4), hard vs soft routing
  Table 4  R matrix (PSNR, stage × task) per variant
  Table 5  QAT: FP32 / PTQ / QAT ACC and BWT (if runs/<run>/qat/results.json exists)
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from metrics import cl_summary


def ms(vals, prec=2):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.{prec}f}"
    return f"{np.mean(vals):.{prec}f} ± {np.std(vals, ddof=1):.{prec}f}"


def mean_over_datasets(res, task, key="psnr"):
    if not res or task not in res:
        return None
    return float(np.mean([v[key] for v in res[task].values()]))


def acc(res, tasks, key="psnr"):
    vals = [mean_over_datasets(res, t, key) for t in tasks]
    return float(np.mean(vals)) if res and all(v is not None for v in vals) else None


def load(runs_dir):
    groups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "results.json"))):
        name = os.path.basename(os.path.dirname(p))
        r = json.load(open(p))
        r["name"] = name
        q = os.path.join(os.path.dirname(p), "qat", "results.json")
        r["qat"] = json.load(open(q)) if os.path.exists(q) else None
        groups[re.sub(r"_s\d+$", "", name)].append(r)
    return groups


def table(header, rows):
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    tex = ["\\begin{tabular}{l" + "c" * (len(header) - 1) + "}", "\\toprule",
           " & ".join(header).replace("_", "\\_") + " \\\\", "\\midrule"]
    tex += [" & ".join(r).replace("±", "$\\pm$").replace("_", "\\_") + " \\\\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(md), "\n".join(tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/tables")
    a = ap.parse_args()
    groups = load(a.runs)
    if not groups:
        print("no runs found")
        return
    md_parts, tex_parts = [], []

    def emit(title, header, rows):
        m, t = table(header, rows)
        md_parts.append(f"### {title}\n\n{m}\n")
        tex_parts.append(f"% {title}\n{t}\n")

    # specialist reference: final PSNR per task
    spec = {}
    for g, rs in groups.items():
        if g.startswith("specialist_"):
            t = g[len("specialist_"):]
            spec[t] = float(np.mean([mean_over_datasets(r["history"][-1]["oracle"], t) for r in rs]))

    cl_groups = {g: rs for g, rs in groups.items() if not g.startswith("specialist_")}
    tasks = max((rs[0]["eval_tasks"] for rs in cl_groups.values()), key=len, default=[])
    datasets = []
    for rs in cl_groups.values():
        for t in tasks:
            for d in rs[0]["history"][-1]["oracle"].get(t, {}):
                if (t, d) not in datasets:
                    datasets.append((t, d))

    # ---- Table 1
    rows = []
    for g, rs in sorted(cl_groups.items()):
        row = [g]
        for t, d in datasets:
            ps = [r["history"][-1]["oracle"].get(t, {}).get(d, {}).get("psnr") for r in rs]
            ss = [r["history"][-1]["oracle"].get(t, {}).get(d, {}).get("ssim") for r in rs]
            row.append(f"{ms(ps)} / {ms(ss, 4)}")
        sums = [r["summary"] if "BWT" in r["summary"] else cl_summary([r["R"][-1]]) for r in rs]
        row += [ms([s["ACC"] if not isinstance(s["ACC"], list) else float(np.mean(s["ACC"])) for s in sums]),
                ms([s.get("BWT") for s in sums]), ms([s.get("FGT") for s in sums])]
        if spec and all(t in spec for t in tasks):
            drops = [float(np.mean([mean_over_datasets(r["history"][-1]["oracle"], t) - spec[t] for t in tasks]))
                     for r in rs]
            row.append(ms(drops))
        else:
            row.append("–")
        row.append(str(len(rs)))
        rows.append(row)
    header = ["method"] + [f"{d} PSNR/SSIM" for _, d in datasets] + ["ACC", "BWT", "FGT", "Δ spec.", "seeds"]
    emit("Final performance after the last task (oracle routing)", header, rows)

    # ---- Table 2: routing
    rows = []
    for g, rs in sorted(cl_groups.items()):
        h = [r["history"][-1] for r in rs]
        if not any(x.get("inferred") for x in h):
            continue
        def racc(key):
            out = []
            for x in h:
                res = x.get(key) or {}
                hits = [v["task_acc"] for t in tasks for v in res.get(t, {}).values() if v.get("task_acc") is not None]
                out.append(float(np.mean(hits)) if hits else None)
            return out
        rows.append([g, ms([acc(x["oracle"], tasks) for x in h]), ms([acc(x.get("inferred"), tasks) for x in h]),
                     ms([acc(x.get("ncm"), tasks) for x in h]), ms([acc(x.get("soft"), tasks) for x in h]),
                     ms(racc("inferred"), 3), ms(racc("ncm"), 3)])
    if rows:
        emit("Task-free inference (ACC, PSNR dB)", ["method", "oracle", "head (hard)", "NCM (hard)", "head (soft)",
                                                    "head routing acc.", "NCM routing acc."], rows)

    # ---- Table 3: compound
    comp = sorted({d for rs in cl_groups.values() for r in rs for d in r["history"][-1]["oracle"].get("compound", {})})
    if comp:
        rows = []
        for g, rs in sorted(cl_groups.items()):
            h = [r["history"][-1] for r in rs]
            row = [g]
            for d in comp:
                for key in ("oracle", "soft"):  # "oracle" on compound sets = hard FDI routing
                    row.append(ms([(x.get(key) or {}).get("compound", {}).get(d, {}).get("psnr") for x in h]))
            rows.append(row)
        emit("Held-out compound degradations (PSNR dB)", ["method"] + [f"{d} {k}" for d in comp for k in ("hard", "soft")], rows)

    # ---- Table 4: R matrices
    for g, rs in sorted(cl_groups.items()):
        R = np.mean([np.asarray(r["R"], float) for r in rs], 0)
        order = rs[0]["eval_tasks"]
        if R.shape[0] < 2:
            continue
        rows = [[f"after {rs[0]['history'][i]['tasks'][0]}"] + [f"{v:.2f}" if j <= i else f"({v:.2f})" for j, v in enumerate(R[i])]
                for i in range(R.shape[0])]
        emit(f"R matrix: {g} (PSNR dB; parentheses = task not trained yet)", ["stage"] + order, rows)

    # ---- Table 5: QAT
    rows = []
    for g, rs in sorted(cl_groups.items()):
        qs = [r["qat"] for r in rs if r.get("qat")]
        if not qs:
            continue
        def qa(k):
            return [acc(q.get(k), tasks) for q in qs]
        rows.append([g, ms(qa("fp32")), ms(qa("ptq")), ms(qa("qat")),
                     ms([q.get("bwt_fp32") for q in qs]), ms([q.get("bwt_qat", q.get("bwt_ptq")) for q in qs]),
                     ms([100 * q.get("int8_weight_coverage", float("nan")) for q in qs], 1)])
    if rows:
        emit("INT8 (ACC PSNR dB, BWT dB)", ["method", "FP32", "PTQ", "QAT", "BWT FP32", "BWT INT8", "INT8 weights %"], rows)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out + ".md", "w").write("\n".join(md_parts))
    open(a.out + ".tex", "w").write("% needs \\usepackage{booktabs}\n" + "\n".join(tex_parts))
    print("\n".join(md_parts))
    print(f"wrote {a.out}.md and {a.out}.tex")


if __name__ == "__main__":
    main()
