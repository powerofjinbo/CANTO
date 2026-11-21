"""Where Canto loses the HNL daughters: the per-step loss breakdown of Fig. 3(a) of paper 1 for Canto and for acorn
with LM, on the HNL validation events (validation only; diagnostic, nothing is chosen on it).

Fig. 3(a) of paper 1 (paper1_v11/figures/fig03_bars.csv) shows, for the standard tracker (acorn (ttbar) with connected
components, CC, at its working cut) and the HNL-daughter targets of the 1,999 HNL validation events in eight |d0|
bins, how many targets each step loses. The same three steps are read out here for

    acorn (ttbar), CC   the standard tracker (regression check against the paper figure)
    acorn (HNL), CC     the same pipeline trained on HNL events, at its working cut
    Canto (ttbar), LM   the tracker of the paper at its frozen validation setting (results/final_selection_v1.json)
    Canto (HNL), LM     every stage trained on the two labeled HNL samples
    acorn (ttbar), LM   acorn's graph and classifier with Canto's builder
    acorn (HNL), LM

LM = the MWS-layer followed by learned merging (hnl.trackbuild.learned_merge), tau per seed as chosen on validation.

Per particle, from the stored validation scores of the run (RUN/gnn_scores/valset.tar; no model is run):
  ceiling    DM match after CC on every true edge of the run's graph (a perfect classifier; the "CC ceiling of the
             graph"). The graph is the one the classifier scored: acorn's metric-learning graph, or for Canto that
             graph plus the triplet legs.
  found      DM match of the tracks of the builder at the working point.
  connected  some track of the builder holds more than half of the particle's hits (DM without the purity condition).
  edges_s0   DM match after CC on the true edges kept by the builder's edge cut (CC: the working cut; LM: s0 = 0.2 of
             the MWS-layer). Diagnostic only.

Classes (`cls`), exclusive, the first that applies:
  found
  graph    not found, and not connected by the true edges of the graph (ceiling false)
  missed   not found, ceiling true, and no track holds more than half of its hits (the track is split)
  merges   not found, ceiling true, one track holds more than half of its hits but too many hits of other particles
A particle that the builder finds although the ceiling counts it as lost in the graph is "found"; these are counted
(found_not_ceiling). The split of `missed` into missed_edges (the true edges above the builder's edge cut do not
connect it either) and missed_builder (they do: the builder split it) is a diagnostic and not part of the figure.

Paper convention ("levels"). Fig. 3(a) is drawn from three levels per bin (hnl.eval.headroom, connectivity variant):
graph = n - k_ceiling, missed true edges = k_ceiling - k_connected, merges = k_connected - k_found. The two
conventions give the same total. They differ by the particles with ceiling false and connected true: such a particle
counts as a graph loss in the levels and lowers "missed true edges" by one, whether it is found or not. The columns
lv_graph, lv_missed, lv_merges hold the level convention; for the standard tracker they must equal the pooled counts
of paper1_v11/figures/fig03_bars.csv (regression check 1).

Populations: the HNL-daughter targets of the HNL validation events ("all HNL daughters", the population of Fig. 3(a))
in the eight |d0| bins of hnl.eval.headroom, and the displaced daughters as a whole (hnl.eval.baseline_val.populations
"primary": |d0| >= 10 mm or r_v >= 100 mm).

Checks (the summary stops on a mismatch):
  1. standard tracker: every pooled and per-seed count of panel (a) of paper1_v11/figures/fig03_bars.csv, in the level
     convention;
     ceiling, found and connected of every particle equal the stored RUN/eval_val tables;
  2. every LM model: found of every particle equals the stored validation grid at the working point; the counts per
     bin and for the displaced daughters equal results/training_sample_val_3seed_v1.csv; Canto (ttbar): the working
     points equal results/final_selection_v1.json;
  3. every run: the ceiling equals the stored RUN/eval_val/ceiling_particle.parquet.

    python -m hnl.eval.canto_losses tasks                                   (the runs to process, one per line)
    python -m hnl.eval.canto_losses build --task K --out DIR --tmp LOCAL [--workers N] [--max-events N]
    python -m hnl.eval.canto_losses summarize --out DIR --results results --name canto_losses_val_v1
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import tarfile
import time

import numpy as np
import pandas as pd

from hnl.eval import bootstrap_tables as BT
from hnl.eval import round4_checks as R4

REPO = BT.REPO
DATA = BT.DATA
OUT = os.path.join(DATA, "runs", "feedback_r5", "canto_losses_val_v1")
BASE = os.path.join(DATA, "runs", "baseline_v1")
WP = "results/working_points_v1.json"
SELECTION = "results/final_selection_v1.json"
BOARD = "results/training_sample_val_3seed_v1.csv"
FIG03 = "paper1_v11/figures/fig03_bars.csv"
SEEDS = ("0", "1", "2")
STD = "acorn (ttbar), CC"
CANTO = "Canto (ttbar), LM"
# (name, builder, how the working points are found)
MODELS = (
    (STD, "CC", ("cc", "i")),
    ("acorn (HNL), CC", "CC", ("cc", "ii")),
    (CANTO, "LM", ("board", "Canto-L (ttbar)")),
    ("Canto (HNL), LM", "LM", ("board", "Canto-L (HNL)")),
    ("acorn (ttbar), LM", "LM", ("board", "acorn (ttbar)")),
    ("acorn (HNL), LM", "LM", ("board", "acorn (HNL)")),
)
D0_EDGES = (0.0, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, np.inf)          # hnl.eval.headroom BINS["d0"]
D0_NAMES = ("d0_0_0.1", "d0_0.1_1", "d0_1_3", "d0_3_10", "d0_10_30", "d0_30_100", "d0_100_300", "d0_300_inf")
D0_TEXT = ("0-0.1 mm", "0.1-1 mm", "1-3 mm", "3-10 mm", "10-30 mm", "30-100 mm", "100-300 mm", ">= 300 mm")
CLASSES = ("found", "graph", "missed", "merges")
FIG_SERIES = {"graph": "graph", "missed edges": "missed", "merges": "merges"}   # fig03_bars.csv series -> class
COUNT_COLS = ["n", "found", "graph", "missed", "merges", "found_not_ceiling", "graph_connected", "missed_edges",
              "missed_builder", "k_ceiling", "k_connected", "lv_graph", "lv_missed", "lv_merges"]


def _repo(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def slug(model: str) -> str:
    return model.replace(" (", "_").replace("), ", "_").replace(")", "").replace(" ", "_").lower()


# --------------------------------------------------------------------------------------------------------------------
# working points


def merge_model(run: str, fam: str, grid: str) -> str:
    """The learned-merging model behind a run's stored LM validation rows: the grid's provenance or the frozen
    selection (hnl.eval.round4_checks.merge_model); for a grid RUN/trackbuild_val_lmerge without either, the run's
    own RUN/lmerge/model.joblib (slurm/stage1/learned_merge.sbatch trains that model and writes that grid with it).
    The summary checks the rebuilt match of every particle against the stored grid."""
    try:
        return R4.merge_model(run, fam, grid)
    except SystemExit:
        own = os.path.join(run, "lmerge", "model.joblib")
        if grid == "trackbuild_val_lmerge" and os.path.exists(own):
            return own
        raise


def specs() -> list[dict]:
    """One entry per model and seed: run, builder family and parameter, merge model, the stored validation rows."""
    wp = json.load(open(_repo(WP)))["cuts_at_FR_star"]
    board = pd.read_csv(_repo(BOARD))
    board = board[board.family == "mwslmerge"]
    sel = json.load(open(_repo(SELECTION)))["pipelines"]["l1tl"]["seeds"]
    out = []
    for name, builder, (kind, key) in MODELS:
        for s in SEEDS:
            if kind == "cc":
                cut = float(wp[key][s])
                run = os.path.join(BASE, f"arm{key}_s{s}")
                sp = {"family": "cc", "param": cut, "stored_wp": f"CC {cut:.2f}", "merge_model": None,
                      "edge_cut": cut, "source": f"{WP} cuts_at_FR_star arm {key}"}
            else:
                r = board[(board.model == key) & (board.seed == int(s))]
                if len(r) != 1:
                    raise SystemExit(f"{BOARD}: {len(r)} rows for {key} seed {s}")
                run, wpt = r.run.iloc[0], r.working_point.iloc[0]
                fam, params, grid = R4.parse_wp(wpt)
                if fam != "mwslmerge":
                    raise SystemExit(f"{BOARD}: {key} seed {s} is not LM ({wpt})")
                sp = {"family": fam, "param": float(params), "stored_wp": wpt,
                      "merge_model": merge_model(run, fam, grid), "edge_cut": R4.edge_cut(fam, float(params)),
                      "source": f"{BOARD} ({key}, mwslmerge)"}
                if name == CANTO:                     # the frozen selection of the paper
                    f = sel[s]
                    if (os.path.normpath(f["run"]) != os.path.normpath(run) or f["config"] != f"{fam},{params}"
                            or os.path.normpath(f["merge_model"]) != os.path.normpath(sp["merge_model"])):
                        raise SystemExit(f"{name} seed {s}: {wpt} of {run} is not the frozen selection {f}")
                    sp["source"] += f"; = {SELECTION} pipelines.l1tl"
            out.append({"model": name, "builder": builder, "seed": int(s), "run": os.path.normpath(run),
                        "key": f"{slug(name)}_s{s}", **sp})
    return out


def tasks() -> list[dict]:
    """The specs grouped by run (one extraction of the run's validation scores serves every spec on it)."""
    by_run: dict = {}
    for sp in specs():
        by_run.setdefault(sp["run"], []).append(sp)
    return [{"run": run, "specs": sps} for run, sps in by_run.items()]


# --------------------------------------------------------------------------------------------------------------------
# build: per particle, per spec


def event_rows(path: str, sps: list[dict], geo: dict, mergers: dict, keep: np.ndarray) -> pd.DataFrame:
    """One scored validation graph: the rows of the particles in `keep` for every spec."""
    from acorn.utils.loading_utils import load_pyg
    from hnl.eval import matching as M
    from hnl.eval.trackbuild_val import load_scored
    from hnl.graphs import ceiling as C
    eid, pid, r, phi, z, _R, a, b, sc = load_scored(path)
    n = len(pid)
    if len(geo["layer_id"]) != n:
        raise ValueError(f"{path}: {len(geo['layer_id'])} hits in the hits table, {n} in the graph")
    g = load_pyg(path[:-3], weights_only=False)
    te = C.unordered_unique(g.track_edges.numpy())
    is_true = np.isin(a.astype(np.int64) * n + b, te[0] * n + te[1])     # as hnl.eval.evaluate_gnn
    s64 = sc.astype(np.float64)
    fm = M.FastMatch(pid)
    ceil = fm.match(C.cc_labels(n, np.stack([a[is_true], b[is_true]])))["dm_matched"]
    sel = np.isin(fm.particles, keep)
    out = []
    for sp in sps:
        if sp["family"] == "cc":
            lab = C.cc_labels(n, np.stack([a[s64 > sp["param"]], b[s64 > sp["param"]]]))
        else:
            lab = R4.build_labels(sp["family"], sp["param"], n, a, b, sc, r, phi, z, geo, mergers[sp["merge_model"]])
        lab = np.asarray(lab)
        m = fm.match(lab)
        k0 = is_true & (s64 > sp["edge_cut"])
        e0 = fm.match(C.cc_labels(n, np.stack([a[k0], b[k0]])))["dm_matched"]
        # the best track of every particle: shared hits, its size, and the number of tracks that hold its hits
        best = np.zeros(fm.P, dtype=np.int64)
        size = np.zeros(fm.P, dtype=np.int64)
        ntrk = np.zeros(fm.P, dtype=np.int64)
        pt_, pp_, ns = m["pair_t"], m["pair_p"], m["pair_shared"]
        if len(pp_):
            on = lab >= 0
            n_track = np.bincount(np.unique(lab[on], return_inverse=True)[1])
            order = np.lexsort((-ns, pp_))
            first = np.r_[True, pp_[order][1:] != pp_[order][:-1]]
            bk = order[first]
            best[pp_[bk]] = ns[bk]
            size[pp_[bk]] = n_track[pt_[bk]]
            np.add.at(ntrk, pp_, 1)
        if not np.array_equal(2 * best > fm.n_particle, m["connected"]):
            raise ValueError(f"{path}: best-track bookkeeping differs from FastMatch.connected")
        out.append(pd.DataFrame({
            "key": sp["key"], "event_id": np.int64(eid), "particle_id": fm.particles[sel].astype(np.int64),
            "n_hits_graph": fm.n_particle[sel], "ceiling": ceil[sel], "found": m["dm_matched"][sel],
            "connected": m["connected"][sel], "edges_s0": e0[sel], "best_shared": best[sel],
            "best_track_size": size[sel], "n_tracks": ntrk[sel]}))
    return pd.concat(out, ignore_index=True)


def _chunk(args):
    paths, sps, geo, keep = args
    from hnl.trackbuild.learned_merge import Merger
    mergers = {sp["merge_model"]: Merger(sp["merge_model"]) for sp in sps if sp["merge_model"]}
    res = []
    for p in paths:
        e = int(os.path.basename(p).split("event")[-1].split(".")[0])
        res.append(event_rows(p, sps, geo[e], mergers, keep[e]))
    return pd.concat(res, ignore_index=True)


def build(task_index: int, out: str, tmp: str, workers: int, max_events: int | None) -> None:
    """One run: extract the HNL validation graphs of RUN/gnn_scores/valset.tar (an explicit list of validation event
    ids; nothing else is read) and write one per-particle table per spec."""
    from hnl.geometry.resolution import hit_geometry
    t0 = time.time()
    task = tasks()[task_index]
    run, sps = task["run"], task["specs"]
    done = [os.path.join(out, sp["key"] + ".parquet") for sp in sps]
    if any(os.path.exists(p) for p in done) and max_events is None:
        raise SystemExit(f"{[p for p in done if os.path.exists(p)]} exist: refusing to overwrite")
    events = BT.expected_events(R4.HNL_SETS)                       # the HNL validation events
    if max_events is not None:
        events = events[:max_events]
    part = R4.hnl_val_particles()
    part = part[part.is_target.astype(bool) & part.category.isin(["hnl", "w"]) & part.event_id.isin(events)]
    keep = {int(e): np.zeros(0, dtype=np.int64) for e in events}
    keep.update({int(e): np.sort(g.particle_id.to_numpy(np.int64)) for e, g in part.groupby("event_id")})
    geo = hit_geometry(events)
    tar_path = os.path.join(run, "gnn_scores", "valset.tar")       # validation scores; never the test tar
    want = {f"event{int(e)}.pyg.gz" for e in events}
    d = os.path.join(tmp, f"task{task_index}")
    os.makedirs(d, exist_ok=True)
    paths = []
    with tarfile.open(tar_path, "r") as tar:
        for mem in tar:
            nm = os.path.basename(mem.name)
            if nm in want:
                mem.name = nm
                tar.extract(mem, d, filter="data")
                paths.append(os.path.join(d, nm))
    if len(paths) != len(want):
        raise SystemExit(f"{tar_path}: extracted {len(paths)} of {len(want)} graphs")
    paths.sort()
    print(f"task {task_index}: {run}; {len(paths)} graphs; specs {[sp['key'] for sp in sps]}; "
          f"{time.time() - t0:.0f} s", flush=True)
    chunks = []
    for k in range(0, len(paths), 10):
        ps = paths[k:k + 10]
        es = [int(os.path.basename(p).split("event")[-1].split(".")[0]) for p in ps]
        chunks.append((ps, sps, {e: geo[e] for e in es}, {e: keep[e] for e in es}))
    if workers > 1:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_chunk, chunks))
    else:
        res = [_chunk(c) for c in chunks]
    t = pd.concat(res, ignore_index=True)
    for p in paths:
        os.remove(p)
    os.rmdir(d)
    if max_events is not None:
        out = os.path.join(out, f"partial_{max_events}")
    os.makedirs(out, exist_ok=True)
    for sp in sps:
        f = os.path.join(out, sp["key"] + ".parquet")
        t[t.key == sp["key"]].drop(columns="key").to_parquet(f, index=False)
        meta = {**sp, "valset_tar": tar_path, "n_events": len(paths), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                "host": os.uname().nodename, "git": R4.git_state(), "elapsed_s": round(time.time() - t0, 1),
                "merge_model_sha256": R4._sha256(sp["merge_model"]) if sp["merge_model"] else "",
                "finished_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        json.dump(meta, open(f[:-8] + ".json", "w"), indent=1)
        print(f"wrote {f}", flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


# --------------------------------------------------------------------------------------------------------------------
# summary


def classify(t: pd.DataFrame) -> np.ndarray:
    return np.where(t.found, "found", np.where(~t.ceiling, "graph", np.where(~t.connected, "missed", "merges")))


def count_row(t: pd.DataFrame) -> dict:
    c = t.cls.to_numpy()
    n, kc, kn, kf = len(t), int(t.ceiling.sum()), int(t.connected.sum()), int(t.found.sum())
    row = {"n": n, **{k: int((c == k).sum()) for k in CLASSES},
           "found_not_ceiling": int((t.found & ~t.ceiling).sum()),
           "graph_connected": int(((c == "graph") & t.connected).sum()),
           "missed_edges": int(((c == "missed") & ~t.edges_s0).sum()),
           "missed_builder": int(((c == "missed") & t.edges_s0).sum()),
           "k_ceiling": kc, "k_connected": kn, "lv_graph": n - kc, "lv_missed": kc - kn, "lv_merges": kn - kf}
    if row["found"] + row["graph"] + row["missed"] + row["merges"] != n or row["found"] != kf:
        raise SystemExit("the classes do not add up")
    if row["lv_graph"] + row["lv_missed"] + row["lv_merges"] != n - kf:
        raise SystemExit("the levels do not add up to the lost particles")
    return row


def load(out: str, checks: list) -> tuple[pd.DataFrame, list[dict]]:
    """The per-particle tables of every spec joined with the truth table; identity checks against the stored tables."""
    info = R4.hnl_val_particles()
    info = info[info.is_target.astype(bool)].reset_index(drop=True)
    from hnl.eval.baseline_val import populations
    pops = populations(info)
    cols = ["event_id", "particle_id", "sample", "d0", "rv", "pt", "eta", "n_hits", "n_pixel_hits", "r_first"]
    hnl = info.loc[pops["hnl_targets"], cols].copy()
    hnl["displaced"] = pops["primary"][pops["hnl_targets"]]
    hnl["abs_d0"] = hnl.d0.abs()
    hnl["d0_bin"] = np.array(D0_NAMES, dtype=object)[np.searchsorted(D0_EDGES, hnl.abs_d0.to_numpy(), "right") - 1]
    sps = specs()
    parts, metas = [], []
    for sp in sps:
        f = os.path.join(out, sp["key"] + ".parquet")
        t = pd.read_parquet(f)
        meta = json.load(open(f[:-8] + ".json"))
        for k in ("run", "family", "param", "merge_model", "stored_wp"):
            if meta[k] != sp[k]:
                raise SystemExit(f"{f}: built with {k} = {meta[k]!r}, the spec says {sp[k]!r}")
        metas.append(meta)
        ev = np.unique(t.event_id)
        if t.duplicated(["event_id", "particle_id"]).any():
            raise SystemExit(f"{f}: duplicate particles")
        # 3. the ceiling equals the stored one, particle by particle
        ce = pd.read_parquet(os.path.join(sp["run"], "eval_val", "ceiling_particle.parquet"))
        j = t.merge(ce, on=["event_id", "particle_id"], how="left")
        ok = bool(j.dm_ceiling.notna().all() and (j.dm_ceiling.astype(bool) == j.ceiling).all())
        checks.append((f"{sp['key']}: ceiling = stored eval_val/ceiling_particle.parquet ({len(j)} particles)",
                       int((j.dm_ceiling.astype(bool) != j.ceiling).sum()) if j.dm_ceiling.notna().all() else "missing",
                       0, ok))
        # 1./2. found (and for CC connected) equal the stored validation rows at the working point
        pp, _pe, d = BT.val_tables(sp["run"], sp["stored_wp"])
        pp = pp[pp.event_id.isin(ev)]
        cols_st = ["event_id", "particle_id", "dm_matched"] + (["connected"] if "connected" in pp else [])
        j = t.merge(pp[cols_st].rename(columns={"connected": "connected_st"}), on=["event_id", "particle_id"],
                    how="left")
        bad = int((j.dm_matched.astype("boolean") != j.found).fillna(True).sum())
        checks.append((f"{sp['key']}: found = stored {os.path.basename(d)} at {sp['stored_wp']!r} "
                       f"({len(j)} particles)", bad, 0, bad == 0))
        if "connected_st" in j:
            bad = int((j.connected_st.astype("boolean") != j.connected).fillna(True).sum())
            checks.append((f"{sp['key']}: connected = stored {os.path.basename(d)}", bad, 0, bad == 0))
        j = hnl.merge(t, on=["event_id", "particle_id"], how="left")
        if j.found.isna().any():
            raise SystemExit(f"{f}: {int(j.found.isna().sum())} HNL-daughter targets without a row")
        for c in ("ceiling", "found", "connected", "edges_s0"):
            j[c] = j[c].astype(bool)
        j["cls"] = classify(j)
        j.insert(0, "seed", sp["seed"])
        j.insert(0, "builder", sp["builder"])
        j.insert(0, "model", sp["model"])
        parts.append(j)
    return pd.concat(parts, ignore_index=True), metas


def tables(p: pd.DataFrame) -> pd.DataFrame:
    """Counts per model, seed (and pooled), population and |d0| bin."""
    rows = []
    for (model, builder), g in p.groupby(["model", "builder"], sort=False):
        for seed, gs in [(str(s), g[g.seed == s]) for s in sorted(g.seed.unique())] + [("pooled", g)]:
            n_seeds = gs.seed.nunique()
            for name, text in zip(D0_NAMES, D0_TEXT):
                rows.append({"model": model, "builder": builder, "seed": seed, "n_seeds": n_seeds,
                             "population": "all HNL daughters", "bin": name, "bin_text": text,
                             **count_row(gs[gs.d0_bin == name])})
            rows.append({"model": model, "builder": builder, "seed": seed, "n_seeds": n_seeds,
                         "population": "all HNL daughters", "bin": "all", "bin_text": "all", **count_row(gs)})
            rows.append({"model": model, "builder": builder, "seed": seed, "n_seeds": n_seeds,
                         "population": "displaced daughters", "bin": "all",
                         "bin_text": "|d0| >= 10 mm or r_v >= 100 mm", **count_row(gs[gs.displaced])})
    t = pd.DataFrame(rows)
    for c in ("found", "graph", "missed", "merges", "lv_graph", "lv_missed", "lv_merges"):
        t[c + "_pct"] = 100.0 * t[c] / t.n
    t["lost_pct"] = 100.0 * (t.n - t.found) / t.n
    return t


def regression(t: pd.DataFrame, checks: list) -> None:
    """Checks 1 and 2 of the docstring on the count table."""
    fig = pd.read_csv(_repo(FIG03))
    seg = fig[(fig.kind == "segment") & (fig.panel == "a")]
    std = t[(t.model == STD) & (t.population == "all HNL daughters")].set_index(["seed", "bin"])
    for series, cls in FIG_SERIES.items():
        for name in D0_NAMES:
            r = seg[(seg.series == series) & (seg.bin == name)]
            if len(r) != 1:
                raise SystemExit(f"{FIG03}: {len(r)} rows for {series} {name}")
            r = r.iloc[0]
            mine = std.loc[("pooled", name)]
            checks.append((f"Fig. 3(a) {series} {name}: pooled count / chances", f"{mine['lv_' + cls]}/{mine['n']}",
                           f"{int(r.k_pooled)}/{int(r.n_pooled)}",
                           int(mine["lv_" + cls]) == int(r.k_pooled) and int(mine["n"]) == int(r.n_pooled)))
            for s in SEEDS:
                m = std.loc[(s, name)]
                want = float(r[f"value_s{s}"]) / 100.0 * float(r.n_per_seed)
                checks.append((f"Fig. 3(a) {series} {name} seed {s}: count", int(m["lv_" + cls]), round(want, 3),
                               abs(m["lv_" + cls] - want) < 2e-3 and int(m["n"]) == int(r.n_per_seed)))
    tot = fig[(fig.kind == "bar total") & (fig.panel == "a")].set_index("bin")
    for name in D0_NAMES:
        mine = std.loc[("pooled", name)]
        checks.append((f"Fig. 3(a) lost {name}: pooled count", int(mine.n - mine.found), int(tot.loc[name].k_pooled),
                       int(mine.n - mine.found) == int(tot.loc[name].k_pooled)))
    # 2. the LM models against the stored board
    board = pd.read_csv(_repo(BOARD))
    board = board[board.family == "mwslmerge"]
    for name, builder, (kind, key) in MODELS:
        if kind != "board":
            continue
        for s in SEEDS:
            r = board[(board.model == key) & (board.seed == int(s))].iloc[0]
            m = t[(t.model == name) & (t.seed == s)]
            a = m[m.population == "all HNL daughters"].set_index("bin")
            mine = [int(a.loc[b].found) for b in D0_NAMES] + [int(a.loc["all"].found)]
            want = [int(r[f"{b}_k"]) for b in D0_NAMES] + [int(r.eff_hnl_targets_k)]
            ns = [int(a.loc[b].n) for b in D0_NAMES] == [int(r[f"{b}_n"]) for b in D0_NAMES]
            checks.append((f"{name} seed {s}: found per |d0| bin and in all = {BOARD}", mine, want,
                           mine == want and ns))
            dd = m[m.population == "displaced daughters"].iloc[0]
            checks.append((f"{name} seed {s}: displaced daughters found / chances = {BOARD}",
                           f"{int(dd.found)}/{int(dd.n)}", f"{int(r.eff_primary_k)}/{int(r.eff_primary_n)}",
                           int(dd.found) == int(r.eff_primary_k) and int(dd.n) == int(r.eff_primary_n)))


def md_table(t: pd.DataFrame, model: str, seed: str) -> list[str]:
    g = t[(t.model == model) & (t.seed == seed)]
    L = ["| abs(d0) | chances | found | lost | graph | missed true edges | merges | found, ceiling lost | "
         "lost [%] | graph [%] | missed [%] | merges [%] |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in g.itertuples():
        lab = r.bin_text if r.population == "all HNL daughters" else "displaced daughters"
        lab = "all HNL daughters" if (r.population == "all HNL daughters" and r.bin == "all") else lab
        L.append(f"| {lab} | {r.n} | {r.found} | {r.n - r.found} | {r.graph} | {r.missed} | {r.merges} | "
                 f"{r.found_not_ceiling} | {r.lost_pct:.2f} | {r.graph_pct:.2f} | {r.missed_pct:.2f} | "
                 f"{r.merges_pct:.2f} |")
    return L


def summarize(out: str, results: str, name: str) -> None:
    checks: list = []
    p, metas = load(out, checks)
    t = tables(p)
    regression(t, checks)
    bad = [c for c in checks if not c[3]]
    print(f"{len(checks)} checks, {len(bad)} failed")
    for c in bad:
        print("FAILED:", c[0], "| mine", c[1], "| stored", c[2])
    if bad:
        raise SystemExit("regression checks failed: nothing written")
    os.makedirs(results, exist_ok=True)
    base = os.path.join(results, name)
    t.to_csv(base + ".csv", index=False, float_format="%.6g")
    lost = p[p.cls != "found"].sort_values(["model", "seed", "abs_d0"])
    lcols = ["model", "builder", "seed", "event_id", "particle_id", "sample", "d0", "rv", "displaced", "d0_bin", "pt",
             "eta", "n_hits", "n_pixel_hits", "r_first", "cls", "ceiling", "connected", "edges_s0", "n_hits_graph",
             "best_shared", "best_track_size", "n_tracks"]
    lost[lcols].to_csv(base + "_lost_particles.csv", index=False, float_format="%.6g")
    fnc = p[p.found & ~p.ceiling]
    pooled = t[t.seed == "pooled"]
    names = [m[0] for m in MODELS]
    L = [f"# Where Canto loses the HNL daughters: the three steps of Fig. 3(a), on validation ({name})", "",
         "Generated by `python -m hnl.eval.canto_losses summarize`. HNL validation events only (1,999 events of the two",
         "labeled HNL samples); stored validation scores, tracks rebuilt on CPU at each model's stored working point;",
         "no model is run or trained and nothing is chosen here. Chances are pooled over the seeds.", "",
         "Classes of a lost daughter, the first that applies: **graph** (CC on the true edges of the graph does not",
         "find it), **missed true edges** (no track holds more than half of its hits), **merges** (one track holds",
         "more than half of its hits, with too many hits of other particles). LM = MWS-layer followed by learned",
         "merging; CC = connected components. A daughter that the builder finds although the true edges of the graph",
         "do not connect it is counted as found (column \"found, ceiling lost\").", "",
         "## Summary", ""]

    def row(m, pop, b="all"):
        return pooled[(pooled.model == m) & (pooled.population == pop) & (pooled.bin == b)].iloc[0]

    def sentence(m):
        d, f = row(m, "displaced daughters"), row(m, "all HNL daughters", "d0_300_inf")
        a = row(m, "all HNL daughters")
        return (f"- **{m}** ({d.n_seeds} seeds): loses {d.n - d.found} of the {d.n} chances of the displaced "
                f"daughters ({d.lost_pct:.2f} points; efficiency {d.found_pct:.2f} %): {d.graph} in the graph, "
                f"{d.missed} to missed true edges, {d.merges} to merges ({d.graph_pct:.2f} / {d.missed_pct:.2f} / "
                f"{d.merges_pct:.2f} points). At abs(d0) >= 300 mm it loses {f.n - f.found} of {f.n}: {f.graph} in "
                f"the graph, {f.missed} to missed true edges, {f.merges} to merges. Found although CC on the true "
                f"edges of the graph does not connect them: {a.found_not_ceiling} of the {a.n} chances of all HNL "
                f"daughters ({f.found_not_ceiling} at abs(d0) >= 300 mm).")

    L += [sentence(m) for m in names]
    d = row(STD, "displaced daughters")
    L += ["", f"The paper quotes the standard tracker with the level convention (see below): {d.lv_graph} / "
          f"{d.lv_missed} / {d.lv_merges} chances = {d.lv_graph_pct:.2f} / {d.lv_missed_pct:.2f} / "
          f"{d.lv_merges_pct:.2f} points, of {d.lost_pct:.2f} lost. Its level counts per bin equal Fig. 3(a) of the "
          "paper.", "",
          "## Displaced daughters (abs(d0) >= 10 mm or r_v >= 100 mm), pooled over the seeds", "",
          "| model | seeds | chances | found | efficiency [%] | lost | graph | missed true edges | merges | "
          "graph [pts] | missed [pts] | merges [pts] | levels: graph / missed / merges |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in names:
        r = row(m, "displaced daughters")
        L.append(f"| {m} | {r.n_seeds} | {r.n} | {r.found} | {r.found_pct:.2f} | {r.n - r.found} | {r.graph} | "
                 f"{r.missed} | {r.merges} | {r.graph_pct:.2f} | {r.missed_pct:.2f} | {r.merges_pct:.2f} | "
                 f"{r.lv_graph} / {r.lv_missed} / {r.lv_merges} |")
    L += ["", "## abs(d0) >= 300 mm (19 daughters per seed), pooled over the seeds", "",
          "| model | chances | found | graph | missed true edges | merges | found, ceiling lost |", "|---|---|---|---|---|---|---|"]
    for m in names:
        r = pooled[(pooled.model == m) & (pooled.bin == "d0_300_inf")].iloc[0]
        L.append(f"| {m} | {r.n} | {r.found} | {r.graph} | {r.missed} | {r.merges} | {r.found_not_ceiling} |")
    L += ["", "## Two ways to count, and the paper figure", "",
          "Fig. 3(a) of the paper is drawn from three levels per bin: graph = chances - ceiling, missed true edges =",
          "ceiling - connected, merges = connected - found. A daughter that is not connected by the true edges of the",
          "graph but sits on a track with more than half of its hits counts as a graph loss there and lowers \"missed",
          "true edges\" by one, whether it is found or not. With the exclusive classes used here it is found, or a",
          "graph loss, and \"missed true edges\" is not lowered. The totals agree. Pooled, all HNL daughters:", "",
          "| model | found although ceiling lost | graph losses with a majority track | graph: classes / levels | "
          "missed: classes / levels | merges: classes / levels |", "|---|---|---|---|---|---|"]
    for m in names:
        r = pooled[(pooled.model == m) & (pooled.population == "all HNL daughters") & (pooled.bin == "all")].iloc[0]
        L.append(f"| {m} | {r.found_not_ceiling} | {r.graph_connected} | {r.graph} / {r.lv_graph} | "
                 f"{r.missed} / {r.lv_missed} | {r.merges} / {r.lv_merges} |")
    L += ["", "The level counts of the standard tracker equal the pooled counts of every bar segment of",
          f"`{FIG03}` (checked). The bins where the two ways differ for it:", ""]
    s = pooled[(pooled.model == STD) & (pooled.population == "all HNL daughters") & (pooled.bin != "all")]
    for r in s[(s.graph != s.lv_graph) | (s.missed != s.lv_missed) | (s.merges != s.lv_merges)].itertuples():
        L.append(f"- {r.bin_text}: classes {r.graph} / {r.missed} / {r.merges}, levels (paper) {r.lv_graph} / "
                 f"{r.lv_missed} / {r.lv_merges} of {r.n} (graph / missed true edges / merges)")
    L += ["", "## Missed true edges with LM: classifier or builder (diagnostic)", "",
          "For a daughter in the class \"missed true edges\": do the true edges above the edge cut of the builder",
          "(LM: s0 = 0.2 of the MWS-layer; CC: the working cut) connect it? If not, the classifier lost it; if they",
          "do, the builder split it. Pooled, all HNL daughters and displaced daughters:", "",
          "| model | missed true edges (all) | classifier | builder split | missed true edges (displaced) | "
          "classifier | builder split |", "|---|---|---|---|---|---|---|"]
    for m in names:
        a = pooled[(pooled.model == m) & (pooled.population == "all HNL daughters") & (pooled.bin == "all")].iloc[0]
        d = pooled[(pooled.model == m) & (pooled.population == "displaced daughters")].iloc[0]
        L.append(f"| {m} | {a.missed} | {a.missed_edges} | {a.missed_builder} | {d.missed} | {d.missed_edges} | "
                 f"{d.missed_builder} |")
    L += ["", "## Working points", "", "| model | seed | run | working point | merge model |", "|---|---|---|---|---|"]
    for sp in specs():
        L.append(f"| {sp['model']} | {sp['seed']} | `{sp['run'].replace(DATA, '$HNL_DATA')}` | {sp['stored_wp']} | "
                 f"`{(sp['merge_model'] or '-').replace(DATA, '$HNL_DATA')}` |")
    for m in names:
        L += ["", f"## {m}: pooled over the seeds", ""] + md_table(t, m, "pooled")
        for sd in SEEDS:
            L += ["", f"### {m}, seed {sd}", ""] + md_table(t, m, sd)
    L += ["", "## Checks", "", f"{len(checks)} checks, all passed:", "",
          f"- standard tracker: the pooled and the per-seed counts of the three segments of every bin of `{FIG03}`",
          "  panel (a) and the lost counts, in the level convention;",
          "- every model and seed: the ceiling of every particle equals the stored",
          "  `RUN/eval_val/ceiling_particle.parquet`; found of every particle equals the stored validation table at",
          "  the working point (CC: also connected);",
          f"- every LM model and seed: found per bin, in all and for the displaced daughters equal `{BOARD}`;",
          "- Canto (ttbar): working points = `results/final_selection_v1.json`.", "",
          f"Files: `{name}.csv` (every count, per seed and pooled; classes and levels), `{name}_lost_particles.csv`",
          f"(every lost HNL daughter with its class), `{name}_meta.json` (inputs, jobs, checks). Per-particle tables:",
          f"`{out.replace(DATA, '$HNL_DATA')}` (Slurm jobs {', '.join(sorted({str(m['slurm_job']) for m in metas}))};",
          "`slurm/feedback/canto_losses.sbatch`). Figures in the style of Fig. 3(a):",
          "`docs/feedback/response_r5/fig3a_canto_*.pdf` (`scripts/diagnostics/fig_canto_losses.py`)."]
    open(base + ".md", "w").write("\n".join(L) + "\n")
    meta = {"script": "src/hnl/eval/canto_losses.py", "command": f"python -m hnl.eval.canto_losses summarize --out "
            f"{out.replace(DATA, '$HNL_DATA')} --results {results} --name {name}",
            "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "git": R4.git_state(),
            "split": "validation: the 1,999 HNL validation events (hnl10_100 and hnl15_200, splits/*_val.txt)",
            "per_particle_tables": out.replace(DATA, "$HNL_DATA"),
            "inputs": {"working_points": WP, "selection": SELECTION, "board": BOARD, "paper_figure": FIG03,
                       "sha256": {f: R4._sha256(_repo(f)) for f in (WP, SELECTION, BOARD, FIG03)}},
            "found_not_ceiling_particles": fnc[["model", "seed", "event_id", "particle_id", "d0", "rv", "d0_bin",
                                                "n_hits_graph", "best_shared", "best_track_size"]].to_dict("records"),
            "builds": [{k: (v.replace(DATA, "$HNL_DATA") if isinstance(v, str) else v) for k, v in m.items()}
                       for m in metas],
            "checks": [{"check": c[0], "mine": str(c[1]), "stored": str(c[2]), "ok": bool(c[3])} for c in checks]}
    json.dump(meta, open(base + "_meta.json", "w"), indent=1)
    print("\n".join(L[:40]))
    print(f"wrote {base}.md, .csv, _lost_particles.csv, _meta.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tasks")
    b = sub.add_parser("build")
    b.add_argument("--task", type=int, required=True)
    b.add_argument("--out", default=OUT)
    b.add_argument("--tmp", required=True)
    b.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    b.add_argument("--max-events", type=int, default=None, help="a few events, for a test (written to a subfolder)")
    s = sub.add_parser("summarize")
    s.add_argument("--out", default=OUT)
    s.add_argument("--results", default="results")
    s.add_argument("--name", default="canto_losses_val_v1")
    a = ap.parse_args(argv)
    if a.cmd == "tasks":
        for i, t in enumerate(tasks()):
            print(i, t["run"], ";".join(f"{sp['key']}={sp['family']},{sp['param']}" for sp in t["specs"]))
    elif a.cmd == "build":
        build(a.task, a.out, a.tmp, a.workers, a.max_events)
    else:
        summarize(a.out, a.results, a.name)


if __name__ == "__main__":
    main()
