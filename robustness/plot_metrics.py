"""Plot the primary robustness comparison from retained numeric evaluations."""
import json
from pathlib import Path

from math import isfinite
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/robustness"
OUT = ROOT / "outputs/figures/robustness_metrics.pdf"
RATES = [0, 5, 10, 20, 40]
KEYS = ["0.00", "0.05", "0.10", "0.20", "0.40"]
MODELS = {
    "bert": ("BERT-base", "#355C9A", "token_length_stats"),
    "pixel-base": ("PIXEL-base", "#D46343", "num_text_patches_stats"),
}
TASKS = {
    "sst2": ("SST-2", "validation", "-", "o"),
    "mnli": ("MNLI", "dev_matched", "--", "^"),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
pdf = canvas.Canvas(str(OUT), pagesize=(515, 198))
pdf.setTitle("Typographical robustness: accuracy loss and input length")
audit = []
for model, (model_label, color, length_key) in MODELS.items():
    for task, (task_label, split, linestyle, marker) in TASKS.items():
        path = RUNS / f"{task}_{model}/results_noisy.json"
        records = json.loads(path.read_text())[split]
        acc = [records[k]["accuracy"] for k in KEYS]
        length = [records[k][length_key]["mean"] for k in KEYS]
        assert all(isfinite(v) for v in acc + length)
        assert acc[0] > 0 and length[0] > 0
        loss = [100 * (acc[0] - v) / acc[0] for v in acc]
        growth = [100 * (v / length[0] - 1) for v in length]
        audit.append({"model": model_label, "task": task_label,
                      "source": str(path), "split": split,
                      "typo_rate_percent": RATES,
                      "accuracy": acc, "mean_input_length": length,
                      "relative_accuracy_loss_percent": loss,
                      "relative_length_change_percent": growth})

def text(x, y, value, size=8, align="left", bold=False, color="#263238"):
    pdf.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    pdf.setFillColor(HexColor(color))
    getattr(pdf, {"left": "drawString", "center": "drawCentredString",
                  "right": "drawRightString"}[align])(x, y, str(value))

def marker(x, y, kind, color):
    pdf.setFillColor(HexColor(color)); pdf.setStrokeColor(white)
    pdf.setLineWidth(.45); pdf.setDash()
    if kind == "o":
        pdf.circle(x, y, 2.3, stroke=1, fill=1)
    else:
        p = pdf.beginPath(); p.moveTo(x, y + 2.8)
        p.lineTo(x + 2.8, y - 2.1); p.lineTo(x - 2.8, y - 2.1); p.close()
        pdf.drawPath(p, stroke=1, fill=1)

for col, (x0, ylo, yhi, ticks, title, ylabel, key) in enumerate([
    (42, -3, 33, [0, 10, 20, 30], "(a) Input length change",
     "Token / patch length change (%)", "relative_length_change_percent"),
    (311, -1, 27, [0, 5, 10, 15, 20, 25], "(b) Accuracy loss",
     "Relative accuracy loss (%)", "relative_accuracy_loss_percent"),
]):
    x = lambda v: x0 + v / 40 * 168
    y = lambda v: 35 + (v - ylo) / (yhi - ylo) * 115
    text(x0, 161, title, size=10, bold=True)
    for tick in ticks:
        pdf.setStrokeColor(HexColor("#D9D9D9")); pdf.setLineWidth(.5)
        pdf.line(x0, y(tick), x0 + 168, y(tick))
        text(x0 - 5, y(tick) - 2.7, tick, align="right")
    pdf.setStrokeColor(HexColor("#263238")); pdf.setLineWidth(.7)
    pdf.line(x0, 35, x0 + 168, 35); pdf.line(x0, 35, x0, 150)
    for rate in RATES:
        pdf.line(x(rate), 35, x(rate), 32)
        text(x(rate), 22, rate, align="center")
    text(x0 + 84, 6, "Typo rate (%)", size=9, align="center")
    pdf.saveState(); pdf.translate(x0 - 28, 92.5); pdf.rotate(90)
    text(0, 0, ylabel, size=9, align="center"); pdf.restoreState()
    for record in audit:
        model = next(v for v in MODELS.values() if v[0] == record["model"])
        task = next(v for v in TASKS.values() if v[0] == record["task"])
        color, linestyle, kind = model[1], task[2], task[3]
        pdf.setStrokeColor(HexColor(color)); pdf.setLineWidth(1.5)
        pdf.setDash([4, 2] if linestyle == "--" else [])
        p = pdf.beginPath()
        for i, (rate, value) in enumerate(zip(RATES, record[key])):
            (p.moveTo if i == 0 else p.lineTo)(x(rate), y(value))
        pdf.drawPath(p)
        for rate, value in zip(RATES, record[key]): marker(x(rate), y(value), kind, color)
        if key == "relative_accuracy_loss_percent":
            text(x(40) + 5, y(record[key][-1]) - 2.6,
                 f"{record[key][-1]:.1f}%", color=color)

for start, label, color, style, kind in [
    (74, "BERT-base", "#355C9A", "-", None),
    (184, "PIXEL-base", "#D46343", "-", None),
    (300, "SST-2", "#555555", "-", "o"),
    (390, "MNLI", "#555555", "--", "^"),
]:
    pdf.setStrokeColor(HexColor(color)); pdf.setLineWidth(1.5)
    pdf.setDash([4, 2] if style == "--" else [])
    pdf.line(start, 184, start + 22, 184)
    if kind: marker(start + 11, 184, kind, color)
    text(start + 28, 181, label, size=9, color=color)
pdf.showPage(); pdf.save()
OUT.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
for record in audit:
    print(record["model"], record["task"], "40% typo: loss=",
          round(record["relative_accuracy_loss_percent"][-1], 4),
          "length change=", round(record["relative_length_change_percent"][-1], 4))
print(OUT)
