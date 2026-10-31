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
  Table 5b INT8 after every task (qat.continual snapshot / quantized): ACC, BWT, INT8 R matrix
  Table 6  comparison with external methods ([2]-[6], all-in-one models, specialists):
           scored outputs from runs/external/*/results.json (eval_external.py) and numbers
           copied from papers (`reported` in configs/external_baselines.yaml, marked *),
           next to our rows (proposed, icvgip_*, specialists)
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


def load(runs_dir, val=False):
    """val=False: test runs only; val=True: only runs made with --eval_on val (suffix _val)."""
    groups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "results.json"))):
        name = os.path.basename(os.path.dirname(p))
        if name.endswith("_val") != val:
            continue
        name = name[:-4] if val else name
        r = json.load(open(p))
        r["name"] = name
        q = os.path.join(os.path.dirname(p), "qat", "results.json")
        r["qat"] = json.load(open(q)) if os.path.exists(q) else None
        groups[re.sub(r"_s\d+$", "", name)].append(r)
    return groups


def external_rows(runs_dir, registry):
    """[(label, {task: {dataset: metrics}}, meta)] from eval_external.py outputs + reported numbers."""
    import yaml
    reg = yaml.safe_load(open(registry)).get("methods", {}) if os.path.exists(registry) else {}
    rows, scored = [], set()
    for p in sorted(glob.glob(os.path.join(runs_dir, "external", "*", "results.json"))):
        r = json.load(open(p))
        flat = {d: v for ds in r["results"].values() for d, v in ds.items()}
        rows.append((r["method"], flat, reg.get(r["method"], r.get("meta", {}))))
        scored.add(r["method"])
    for name, meta in reg.items():
        rep = meta.get("reported")
        if rep and name not in scored:
            rows.append((name + "*", rep, meta))
    return rows


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
    ap.add_argument("--registry", default="configs/external_baselines.yaml")
    ap.add_argument("--val", action="store_true", help="tabulate the --eval_on val runs (hyperparameter search)")
    ap.add_argument("--ours", nargs="*", default=["proposed", "icvgip_sr", "icvgip_cl", "joint", "specialist_sr",
                                                  "specialist_denoise", "specialist_deblur", "specialist_jpeg"],
                    help="our variants to show in the external comparison")
    a = ap.parse_args()
    groups = load(a.runs, a.val)
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

    # ---- Table 5b: INT8 after every task (train.py qat.continual)
    rows = []
    for g, rs in sorted(cl_groups.items()):
        qm = rs[0].get("qat_continual", "none")
        if qm == "snapshot":
            fp_s = [r["summary"] for r in rs]
            q_s = [r["summary_int8"] for r in rs]
        elif qm == "quantized":
            fp_s, q_s = [{} for _ in rs], [r["summary"] for r in rs]
        else:
            continue
        rows.append([g, qm, ms([x.get("ACC") for x in fp_s]), ms([x.get("ACC") for x in q_s]),
                     ms([x.get("BWT") for x in fp_s]), ms([x.get("BWT") for x in q_s]), ms([x.get("FGT") for x in q_s])])
    if rows:
        emit("INT8 after every task (snapshot: QAT copy of the FP32 model; quantized: INT8 from task 1 on)",
             ["method", "mode", "ACC FP32", "ACC INT8", "BWT FP32", "BWT INT8", "FGT INT8"], rows)
        for g, rs in sorted(cl_groups.items()):
            if rs[0].get("R_int8"):
                R = np.mean([np.asarray(r["R_int8"], float) for r in rs], 0)
                order = rs[0]["eval_tasks"]
                emit(f"INT8 R matrix (snapshot): {g}", ["stage"] + order,
                     [[f"after {rs[0]['history'][i]['tasks'][0]}"] + [f"{v:.2f}" if j <= i else f"({v:.2f})"
                                                                      for j, v in enumerate(R[i])] for i in range(R.shape[0])])

    # ---- Table 6: external comparison (columns = dataset names)
    ext = external_rows(a.runs, a.registry)
    if ext:
        cols = []
        for _, res, _ in ext:
            for d in res:
                if d not in cols:
                    cols.append(d)
        ours = []
        for g in a.ours:
            if g not in groups:
                continue
            rs = groups[g]
            res = {}
            for d in cols:
                vals = [v for r in rs for ds in r["history"][-1]["oracle"].values() for n, v in ds.items() if n == d]
                if vals:
                    res[d] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]
                              if isinstance(vals[0][k], float) and all(v.get(k) is not None for v in vals)}
            p = rs[0].get("params", {}).get("deployable")
            ours.append((f"ours: {g}", res, {"params_M": p / 1e6 if p else None}))
        metric_keys = []
        for _, res, _ in ext + ours:
            for v in res.values():
                for k in v:
                    if k not in metric_keys and k not in ("n", "task_acc", "routed_to"):
                        metric_keys.append(k)
        rows = []
        for name, res, meta in ext + ours:
            row = [name, "–" if meta.get("params_M") is None else f"{meta['params_M']:.2f}"]
            for d in cols:
                v = res.get(d)
                row.append(" / ".join("–" if not v or v.get(k) is None else
                                      f"{v[k]:.4f}" if k in ("ssim", "lpips") else f"{v[k]:.2f}" for k in metric_keys))
            rows.append(row)
        emit("Comparison with other methods (* = numbers reported in the paper, not re-scored)",
             ["method", "params (M)"] + [f"{d} ({'/'.join(metric_keys)})" for d in cols], rows)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out + ".md", "w").write("\n".join(md_parts))
    open(a.out + ".tex", "w").write("% needs \\usepackage{booktabs}\n" + "\n".join(tex_parts))
    print("\n".join(md_parts))
    print(f"wrote {a.out}.md and {a.out}.tex")


if __name__ == "__main__":
    main()
