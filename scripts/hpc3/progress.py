#!/usr/bin/env python3
"""HPC3: write a live progress page (PROGRESS.html, auto-refresh) and PROGRESS.txt from the chain state, the
training logs and squeue. Run by slurm/hpc3/progress.sbatch in a loop; reads only, writes the two files atomically."""
import argparse, glob, html, json, os, re, subprocess, time

OUT = os.environ.get("HNL_OUT", "/pub/jinboz1/hnl_lrt/hpc3_v1")
CHAIN = os.path.join(OUT, "_chain")
LOGS = "/pub/jinboz1/hnl/logs_hpc3"
TRAIN = [("Canto-L (paid A30)", r"canto_l1tl_(i|ii)_s\d$"), ("Canto-B (free-gpu)", r"canto_l1t_base_(i|ii)_s\d$"),
         ("acorn 60-epoch (free-gpu)", r"ignn60_(i|ii)_s\d$")]
EPOCH = re.compile(r"^\[epoch\] (\d+) step (\d+) t (\d+)s", re.M)


def sh(cmd, timeout=60):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def load(d):
    out = {}
    for p in glob.glob(os.path.join(CHAIN, d, "*.json")):
        try:
            j = json.load(open(p))
            out[j["key"]] = j
        except Exception:
            pass
    return out


def epochs_done(st):
    best = 0
    for jid in st.get("jobs", []):
        for p in glob.glob(os.path.join(LOGS, f"*_{jid}.out")):
            try:
                m = EPOCH.findall(open(p, errors="replace").read())
            except OSError:
                continue
            if m:
                best = max(best, int(m[-1][0]) + 1)
    return best


def collect():
    defs, states = load("defs"), load("state")
    q = {}
    for line in sh("squeue -h -u $USER -o '%i|%T|%P|%M|%l|%j'").splitlines():
        f = line.split("|")
        if len(f) == 6:
            q[f[0]] = f
    bal = ""
    for line in sh("sbank balance statement -a asterenb_lab_gpu").splitlines():
        if line.upper().startswith("ASTERENB_LAB_GPU"):
            bal = line.split()[-1]
    done = sum(1 for k in defs if states.get(k, {}).get("status") == "done")
    groups = []
    for title, pat in TRAIN:
        rows = []
        for k in sorted(k for k in defs if re.match(pat, k)):
            st = states.get(k, {})
            total = int(defs[k]["sbatch"][-1])
            jid = st.get("jobs", [""])[-1] if st.get("jobs") else ""
            js = q.get(jid, [None, ""])[1]
            fin = st.get("status") == "done"
            e = total if fin else min(epochs_done(st), total)
            label = "done" if fin else {"RUNNING": "running", "PENDING": "queued"}.get(js, st.get("status", "not submitted"))
            rows.append((k, e, total, label, q.get(jid, [""] * 4)[3] if js == "RUNNING" else ""))
        groups.append((title, rows))
    running = sorted((f[5][2:] if f[5].startswith("c_") else f[5], f[2], f[3], f[4]) for f in q.values() if f[1] == "RUNNING" and f[5] != "vscode-sshd")
    npend = sum(1 for f in q.values() if f[1] == "PENDING")
    bad = sorted(k for k, s in states.items() if s.get("status") in ("failed", "stopped", "cancelled"))
    try:
        log = open(os.path.join(CHAIN, "log.txt"), errors="replace").read().splitlines()[-12:]
    except OSError:
        log = []
    return dict(t=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), done=done, total=len(defs), groups=groups,
                running=running, npend=npend, bal=bal, bad=bad, log=log)


def bar(n, m, w=30):
    f = int(round(w * n / m)) if m else 0
    return "#" * f + "." * (w - f)


def text(d):
    L = [f"HPC3 progress   {d['t']}", "",
         f"chain steps  [{bar(d['done'], d['total'], 40)}] {d['done']}/{d['total']} ({100 * d['done'] // max(d['total'], 1)}%)",
         f"paid balance {d['bal'] or 'n/a'} SU    running {len(d['running'])}    queued {d['npend']}"]
    if d["bad"]:
        L.append("NEEDS ATTENTION: " + ", ".join(d["bad"]))
    for title, rows in d["groups"]:
        L += ["", title]
        for k, e, m, lab, el in rows:
            L.append(f"  {k:24s} [{bar(e, m)}] {e:2d}/{m} {lab} {el}")
    L += ["", "running jobs"] + [f"  {n:34s} {p:9s} {el:>11s} / {lim}" for n, p, el, lim in d["running"]]
    L += ["", "last chain events"] + ["  " + x[:150] for x in d["log"]]
    return "\n".join(L) + "\n"


def page(d):
    e = html.escape
    def row(name, n, m, lab, extra=""):
        pct = 100 * n / m if m else 0
        return (f'<div class="r"><span class="n">{e(name)}</span><div class="b"><div class="f {e(lab.split()[0])}" style="width:{pct:.1f}%"></div></div>'
                f'<span class="v">{n}/{m} · {pct:.0f}%</span><span class="s">{e(lab)} {e(extra)}</span></div>')
    te = sum(r[1] for _, rows in d["groups"] for r in rows); tm = sum(r[2] for _, rows in d["groups"] for r in rows)
    h = [f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta http-equiv="refresh" content="60">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>HPC3 progress</title><style>
:root{{--bg:#fcfcfb;--fg:#0b0b0b;--mu:#6b6a66;--ln:#e1e0d9;--tr:#eeede8}}
@media(prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--fg:#f0efec;--mu:#a3a29b;--ln:#2c2c2a;--tr:#2c2c2a}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;max-width:900px;margin:0 auto;padding:16px}}
h1{{font-size:18px;font-weight:500;margin:0}}h2{{font-size:14px;font-weight:500;margin:18px 0 6px}}.mu{{color:var(--mu);font-size:12px}}
.k{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:12px 0}}
.k div{{border:1px solid var(--ln);border-radius:8px;padding:10px}}.k b{{display:block;font-size:22px;font-weight:500}}
.r{{display:grid;grid-template-columns:150px minmax(0,1fr) 92px 130px;gap:10px;align-items:center;margin:4px 0;font-size:13px}}
.b{{height:10px;background:var(--tr);border-radius:5px;overflow:hidden}}.f{{height:100%;background:#888780}}
.f.running{{background:#2a78d6}}.f.done{{background:#0ca30c}}.v,.s{{color:var(--mu);font-variant-numeric:tabular-nums}}.v{{text-align:right}}
.w{{color:#d03b3b}}pre{{font-size:12px;white-space:pre-wrap;word-break:break-all;color:var(--mu);margin:0}}
@media(max-width:600px){{.r{{grid-template-columns:110px minmax(0,1fr) 80px}}.s{{display:none}}}}
</style></head><body><h1>HPC3 progress</h1><div class="mu">updated {e(d['t'])} · this page reloads every 60 s</div>
<div class="k"><div>chain steps<b>{d['done']} / {d['total']}</b></div><div>training epochs<b>{te} / {tm}</b></div>
<div>paid balance<b>{e(d['bal'] or 'n/a')} SU</b></div><div>running / queued<b>{len(d['running'])} / {d['npend']}</b></div></div>"""]
    if d["bad"]:
        h.append('<div class="w">Needs attention: ' + e(", ".join(d["bad"])) + "</div>")
    h.append(row("all chain steps", d["done"], d["total"], "running"))
    for title, rows in d["groups"]:
        h.append(f"<h2>{e(title)}</h2>")
        h += [row(k, n, m, lab, el) for k, n, m, lab, el in rows]
    h.append('<div class="mu" style="margin-top:8px">blue = running, grey = queued or waiting, green = done</div><h2>Running jobs</h2><pre>')
    h.append(e("\n".join(f"{n:34s} {p:9s} {el:>11s} / {lim}" for n, p, el, lim in d["running"])))
    h.append("</pre><h2>Last chain events</h2><pre>" + e("\n".join(d["log"])) + "</pre></body></html>")
    return "".join(h)


def write(path, s):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(s)
    os.replace(tmp, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/pub/jinboz1/hnl/hnl-displaced")
    ap.add_argument("--loop", type=int, default=0, help="seconds between updates (0 = once)")
    a = ap.parse_args()
    while True:
        try:
            d = collect()
            write(os.path.join(a.dir, "PROGRESS.txt"), text(d))
            write(os.path.join(a.dir, "PROGRESS.html"), page(d))
        except Exception as ex:
            print("progress: update failed:", repr(ex), flush=True)
        if not a.loop:
            break
        time.sleep(a.loop)
