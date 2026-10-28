"""Generates docs/architecture.svg (model + training/deployment pipeline).

    python scripts/make_arch_diagram.py            # writes docs/architecture.svg
Convert for LaTeX:  rsvg-convert -f pdf docs/architecture.svg > docs/architecture.pdf
"""
import os

W, H = 1500, 960
C = {
    "ink": "#1f2937", "muted": "#6b7280", "line": "#4b5563",
    "box": "#f8fafc", "boxs": "#94a3b8",
    "blue": "#1d4ed8", "bluef": "#eff6ff",
    "orange": "#c2410c", "orangef": "#fff7ed",
    "green": "#047857", "greenf": "#ecfdf5",
    "violet": "#6d28d9", "violetf": "#f5f3ff",
}
out = []


def T(x, y, s, size=13, weight="normal", color=None, anchor="middle", italic=False):
    st = ' font-style="italic"' if italic else ""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color or C["ink"]}" '
               f'text-anchor="{anchor}"{st}>{s}</text>')


def box(x, y, w, h, lines, fill=None, stroke=None, dashed=False, title_color=None, r=8, size=12.5, sw=1.4):
    d = ' stroke-dasharray="6 4"' if dashed else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill or C["box"]}" '
               f'stroke="{stroke or C["boxs"]}" stroke-width="{sw}"{d}/>')
    n = len(lines)
    lh = size + 4
    y0 = y + h / 2 - (n - 1) * lh / 2 + size * 0.35
    for i, l in enumerate(lines):
        bold = i == 0
        T(x + w / 2, y0 + i * lh, l, size if bold else size - 1.5, "600" if bold else "normal",
          (title_color if bold and title_color else (C["ink"] if bold else C["muted"])))


def arrow(pts, color=None, dashed=False, width=1.6, head=True):
    d = ' stroke-dasharray="6 4"' if dashed else ""
    p = " ".join(f"{a},{b}" for a, b in pts)
    mk = f' marker-end="url(#ah-{(color or C["line"]).strip("#")})"' if head else ""
    out.append(f'<polyline points="{p}" fill="none" stroke="{color or C["line"]}" stroke-width="{width}"'
               f' stroke-linejoin="round"{d}{mk}/>')


def circle_plus(cx, cy, r=11):
    out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="white" stroke="{C["line"]}" stroke-width="1.6"/>')
    out.append(f'<path d="M{cx - 6},{cy} H{cx + 6} M{cx},{cy - 6} V{cy + 6}" stroke="{C["line"]}" stroke-width="1.6"/>')


def tag(x, y, s, color, fill):
    w = 7.2 * len(s) + 12
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="18" rx="9" fill="{fill}" stroke="{color}" stroke-width="1"/>')
    T(x + w / 2, y + 13, s, 10.5, "600", color)


# ======================================================================= (a) model
T(30, 34, "(a) RestoreNet — one compact network for SR, denoising, deblurring and JPEG removal",
  17, "700", anchor="start")
T(30, 54, "solid = deployed inference path   ·   dashed = training only   ·   "
  "blue = degradation (FDI) conditioning   ·   orange = per-task LoRA adapter points",
  12, color=C["muted"], anchor="start")

yc, bh = 300, 76            # main path centre line / box height
top = yc - bh / 2

# input
box(30, top, 115, bh, ["LQ image", "H×W×3", "(SR: bicubic-up)"])
# conditioning path
box(185, 128, 280, 64, ["Degradation encoder", "5 conv (stride 2) → mean+std pool → MLP"],
    C["bluef"], C["blue"], title_color=C["blue"])
arrow([(87, top), (87, 160), (183, 160)], C["blue"])
box(505, 136, 160, 48, ["FDI embedding f", "f ∈ ℝ^256, once per image"], "white", C["blue"], title_color=C["blue"])
arrow([(465, 160), (503, 160)], C["blue"])

# training-only heads (top row)
box(505, 70, 160, 44, ["FDI head (linear)", "student → teacher space"], "white", C["muted"], dashed=True)
arrow([(585, 136), (585, 116)], C["muted"], dashed=True)
box(700, 66, 330, 52, ["L_fdi = ‖head(f) − norm(T(LQ) − T(HQ))‖²",
                       "T = frozen DINOv2-S teacher (train-time only, 0 deploy cost)"],
    "white", C["muted"], dashed=True)
arrow([(665, 92), (698, 92)], C["muted"], dashed=True)

box(1065, 128, 230, 64, ["Task head (linear)", "L_task in training; at inference picks", "adapter t̂ — no task label"],
    C["violetf"], C["violet"], title_color=C["violet"])
arrow([(665, 160), (1063, 160)], C["violet"])
T(1180, 214, "routes ↓ which adapter set is active", 11, color=C["violet"], italic=True)

# FDI bus
bus_y = 214
arrow([(585, 184), (585, bus_y), (1050, bus_y)], C["blue"], head=False, width=2)
for xx in (585, 790, 1050):
    arrow([(xx, bus_y), (xx, top - 2)], C["blue"], width=2)
T(700, bus_y - 6, "f conditions every block + decoder", 11, color=C["blue"], italic=True)

# main path
box(185, top, 120, bh, ["Shallow conv", "3×3, 32 ch", "full resolution"])
arrow([(145, yc), (183, yc)])
box(340, top, 130, bh, ["PixelUnshuffle ×4", "+ 1×1 conv", "→ 96 ch, H/4×W/4"])
arrow([(305, yc), (338, yc)])
box(505, top, 165, bh, ["8 × Gated block", "window attn + FFN", "SE-gated by f"], "white", C["ink"], sw=2)
tag(538, top + bh + 6, "LoRA on gates", C["orange"], C["orangef"])
arrow([(470, yc), (503, yc)])
box(705, top, 125, bh, ["LayerNorm", "+ AdaIN-1(f)", "96 ch"])
tag(712, top + bh + 6, "LoRA on γ, β", C["orange"], C["orangef"])
arrow([(670, yc), (703, yc)])
box(865, top, 130, bh, ["1×1 conv +", "PixelShuffle ×4", "→ 32 ch, H×W"])
arrow([(830, yc), (863, yc)])
circle_plus(1020, yc)
arrow([(995, yc), (1007, yc)])
box(1050, top, 110, bh, ["AdaIN-2(f)", "32 ch"])
tag(1052, top + bh + 6, "LoRA on γ, β", C["orange"], C["orangef"])
arrow([(1031, yc), (1048, yc)])
box(1190, top, 100, bh, ["Conv 3×3", "→ 3 ch", "(residual)"])
arrow([(1160, yc), (1188, yc)])
circle_plus(1318, yc)
arrow([(1290, yc), (1305, yc)])
box(1350, top, 120, bh, ["Restored", "image", "H×W×3"], C["greenf"], C["green"], title_color=C["green"])
arrow([(1329, yc), (1348, yc)])

# skips
arrow([(245, top + bh), (245, 388), (1020, 388), (1020, yc + 13)], C["line"])
T(640, 383, "full-resolution skip (carries fine detail the 4× latent cannot)", 11, color=C["muted"], italic=True)
arrow([(87, top + bh), (87, 414), (1318, 414), (1318, yc + 13)], C["line"])
T(700, 409, "global residual: output = LQ + predicted correction", 11, color=C["muted"], italic=True)

# ----- detail insets
iy, ih = 436, 130
out.append(f'<rect x="30" y="{iy}" width="1440" height="{ih}" rx="10" fill="white" stroke="{C["boxs"]}" stroke-dasharray="3 3"/>')
T(42, iy + 20, "Inside a gated block", 13, "700", anchor="start")
bx, by = 42, iy + 42
seq = [("z", 30), ("LN", 42), ("W-MSA", 92), ("⊗ g_a(f)", 76), ("⊕", 26), ("LN", 42), ("FFN 2×", 72), ("⊗ g_f(f)", 76), ("⊕", 26)]
x = bx
for i, (s, w) in enumerate(seq):
    isg = s.startswith("⊗")
    box(x, by, w, 34, [s], C["bluef"] if isg else C["box"], C["blue"] if isg else C["boxs"],
        title_color=C["blue"] if isg else None, size=12)
    if i < len(seq) - 1:
        arrow([(x + w, by + 17), (x + w + 10, by + 17)], width=1.2)
    x += w + 12
T(bx, by + 56, "W-MSA: 8×8 windows, 4 heads; odd blocks shift by 4 px.  Residual adds around each gated branch.",
  11, color=C["muted"], anchor="start")
T(bx, by + 72, "Attention matmul inputs (q, k, v, softmax) carry INT8 quantizers after QAT.", 11,
  color=C["muted"], anchor="start")

sx = 680
T(sx, iy + 20, "SE gate g(f)", 13, "700", anchor="start")
seq = [("f", 26), ("FC", 70), ("ReLU", 46), ("FC", 70), ("σ", 30)]
x = sx
for i, (s, w) in enumerate(seq):
    lora = s == "FC"
    box(x, by, w, 34, [s + (" +BA" if lora else "")], C["orangef"] if lora else C["box"],
        C["orange"] if lora else C["boxs"], title_color=C["orange"] if lora else None, size=12)
    if i < len(seq) - 1:
        arrow([(x + w, by + 17), (x + w + 9, by + 17)], width=1.2)
    x += w + 11
T(sx, by + 56, "σ init ≈ 0.88 (gate starts open)", 11, color=C["muted"], anchor="start")

ax = 1000
T(ax, iy + 20, "AdaIN(x, f)  and  per-task LoRA", 13, "700", anchor="start")
T(ax, by + 14, "y = x + γ(f) · IN(x) + β(f),   [γ, β] = FC(ReLU(FC(f)))", 12, anchor="start")
T(ax, by + 36, "LoRA layer:  W x + b + (α/r) · B_t A_t x", 12, color=C["orange"], anchor="start", weight="600")
T(ax, by + 56, "task 0 trains W; task t adds its own (A_t, B_t), rank 8;", 11, color=C["muted"], anchor="start")
T(ax, by + 72, "B_t = 0 at init so a new task starts from the old behaviour", 11, color=C["muted"], anchor="start")

# ======================================================================= (b) pipeline
py = 606
T(30, py, "(b) Continual training → INT8 quantization → edge deployment", 17, "700", anchor="start")

sy, sh = py + 22, 150
stages = [
    ("Stage 0 · SR ×2/3/4", ["DF2K, bicubic", "train ALL params", "L_rec + L_fdi + L_freq", "+ L_dual + L_task"]),
    ("Stage 1 · Denoise", ["DF2K + BSD400 + WED", "σ ∈ [0, 50]", "freeze base; add LoRA_dn", "train enc. + heads"]),
    ("Stage 2 · Deblur", ["GoPro pairs", "patch 256", "freeze base; add LoRA_db", "+ replay + EWC"]),
    ("Stage 3 · JPEG", ["DF2K + BSD400", "q ∈ [10, 40]", "freeze base; add LoRA_jp", "+ replay + EWC"]),
]
sw_, gap = 200, 28
for i, (title, lines) in enumerate(stages):
    x = 30 + i * (sw_ + gap)
    box(x, sy, sw_, sh, [title] + lines, "white", C["ink"], sw=1.6, size=13)
    if i < 3:
        arrow([(x + sw_, sy + sh / 2), (x + sw_ + gap - 2, sy + sh / 2)], width=2)
T(1470, py, "FP32 · AdamW + cosine LR · EMA weights · one checkpoint per stage", 11.5,
  color=C["muted"], anchor="end")

qx = 30 + 4 * (sw_ + gap) + 10
q_items = [
    ("INT8 PTQ → QAT", ["calibrate observers on", "ALL tasks (own adapter)", "QAT: task-balanced replay,", "KD from FP32, LR 5e-6"]),
    ("ONNX QDQ export", ["one graph per task", "(adapter baked in)", "parity: ORT vs PyTorch", "fake-quant output"]),
    ("Edge deployment", ["TensorRT INT8 engine", "(e.g. Jetson Orin Nano)", "W: per-channel sym.", "A: per-tensor asym."]),
]
qw, qg = 158, 22
arrow([(30 + 3 * (sw_ + gap) + sw_, sy + sh / 2), (qx - 2, sy + sh / 2)], C["green"], width=2)
for i, (title, lines) in enumerate(q_items):
    x = qx + i * (qw + qg)
    box(x, sy, qw, sh, [title] + lines, C["greenf"], C["green"], title_color=C["green"], sw=1.6, size=13)
    if i < 2:
        arrow([(x + qw, sy + sh / 2), (x + qw + qg - 2, sy + sh / 2)], C["green"], width=2)

# replay buffer
ry = sy + sh + 34
out.append(f'<rect x="258" y="{ry}" width="{qx + qw - 258}" height="46" rx="23" fill="{C["violetf"]}" '
           f'stroke="{C["violet"]}" stroke-width="1.4"/>')
T(258 + (qx + qw - 258) / 2, ry + 19, "Replay buffer — ~500 LQ/HQ patches per finished task (reservoir)", 13, "600", C["violet"])
T(258 + (qx + qw - 258) / 2, ry + 36, "25% of every later batch · source of all-task calibration and QAT data",
  11, color=C["muted"])
for i in range(1, 4):
    x = 30 + i * (sw_ + gap) + sw_ / 2
    arrow([(x, ry), (x, sy + sh + 2)], C["violet"])
arrow([(qx + qw / 2, ry), (qx + qw / 2, sy + sh + 2)], C["violet"])
arrow([(30 + sw_ / 2, sy + sh), (30 + sw_ / 2, ry + 23), (256, ry + 23)], C["violet"], dashed=True)
T(36, ry + 40, "filled after each stage", 10.5, color=C["violet"], anchor="start", italic=True)

# evaluation row
ey = ry + 62
box(30, ey, 4 * sw_ + 3 * gap, 66, ["After every stage: evaluate ALL tasks (oracle + inferred routing)",
                                     "R[i][j] PSNR matrix → ACC · BWT · FGT   |   held-out compound set (RQ4)   |   FDI linear probe (RQ2)"],
    C["box"], C["boxs"], size=13)
box(qx, ey, 3 * qw + 2 * qg, 66, ["Report: FP32 vs PTQ vs QAT, per task",
                                  "ΔPSNR / ΔLPIPS · BWT before vs after QAT · params, GFLOPs, latency"],
    C["box"], C["boxs"], size=13)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Inter, Helvetica, Arial, sans-serif">\n'
       '<defs>' + "".join(
           f'<marker id="ah-{c.strip("#")}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
           for c in {C["line"], C["blue"], C["muted"], C["violet"], C["green"], C["orange"]}) + '</defs>\n'
       f'<rect width="{W}" height="{H}" fill="white"/>\n' + "\n".join(out) + "\n</svg>\n")
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
os.makedirs(root, exist_ok=True)
open(os.path.join(root, "architecture.svg"), "w").write(svg)
print("wrote docs/architecture.svg")
