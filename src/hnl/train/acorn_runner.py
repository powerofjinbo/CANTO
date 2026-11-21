"""Run the acorn baseline stages (brief §5f) with explicit data, seeds, checkpoints and run metadata.

Rules (docs/acorn_pitfalls.md, docs/acorn_map.md): per-event files live only on node-local disk; checkpoints and
packed outputs on shared storage; every inference gets an explicit checkpoint and explicit input_dir / stage_dir / data_split /
hard_cuts / undirected; training uses acorn's trainer setup (single-device DDPStrategy, so training events are
reshuffled every epoch; edge classifiers continue that order after a mid-epoch resume, hnl.train.resume_sampler), a
periodic checkpoint every 30 min, and a pre-timeout checkpoint on SIGUSR1.

    python -m hnl.train.acorn_runner stage --arm ii --source fs --what train,valmon --dest $LOCAL/fs
    python -m hnl.train.acorn_runner train --stage ml --arm ii --seed 0 --run-dir $RUN --input-dir $LOCAL/fs
    python -m hnl.train.acorn_runner infer --stage ml --ckpt $CKPT --input-dir $LOCAL/fs --out-dir $LOCAL/ml \\
        [--r-infer 0.1 --knn-infer 1000]
    python -m hnl.train.acorn_runner pack --src $LOCAL/ml --dest $RUN/ml_graphs
"""

from __future__ import annotations

import hnl.paths as _hnl_paths

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import tarfile
import time

import numpy as np
import yaml

from hnl.data.acorn_store import SAMPLE_CODE, event_id, parse_event_id
from hnl.data.splits import read_split

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DERIVED = os.path.join(_hnl_paths.DERIVED)
FS_DIR = os.path.join(DERIVED, "acorn_fs", "v1")
TEMPLATES = {"ml": os.path.join(REPO, "configs", "baseline", "ml_train.yaml"),
             "gnn": os.path.join(REPO, "configs", "baseline", "gnn_train.yaml")}
N_VAL_MONITOR = 500
ACORN_SPLIT = {"train": "trainset", "valmon": "valset", "val": "valset", "val_all": "valset", "test": "testset",
               "test_all": "testset", "external": "testset"}
LABELED = ("ttbar", "hnl10_100", "hnl15_200")


# --------------------------------------------------------------------------------------------- arms and event lists
def arm_events(arm: str, what: str) -> list[tuple[str, int]]:
    """(sample, index) of one arm's events. what: train | valmon (checkpoint monitoring subset) | val (the arm's own
    val mix) | val_all / test_all (the val / test splits of all three labeled samples, whatever the arm) | test |
    external. Arm (iii) train: all ttbar train + HNL train events (hnl10_100 index < 5000, hnl15_200 index >= 5000),
    trimmed to the same count as ttbar (DECISIONS.md 2026-10-03)."""
    comps = {"i": ["ttbar"], "ii": ["hnl10_100", "hnl15_200"], "iii": ["ttbar", "hnl10_100", "hnl15_200"]}[arm]
    if what == "external":
        return [("hnl15_100", int(i)) for i in read_split("hnl15_100", "external")]
    if what in ("val_all", "test_all"):
        return [(s, int(i)) for s in LABELED for i in read_split(s, what[:-4])]
    split = {"train": "train", "valmon": "val", "val": "val", "test": "test"}[what]
    out: list[tuple[str, int]] = []
    if arm == "iii" and what == "train":
        tt = [("ttbar", int(i)) for i in read_split("ttbar", "train")]
        hnl = [("hnl10_100", int(i)) for i in read_split("hnl10_100", "train") if i < 5000] + \
              [("hnl15_200", int(i)) for i in read_split("hnl15_200", "train") if i >= 5000]
        return tt + hnl[:len(tt)]
    per = []
    for s in comps:
        per.append([(s, int(i)) for i in read_split(s, split)])
    if what == "valmon":
        k = N_VAL_MONITOR // len(per)
        per = [p[:k] for p in per]
    for p in per:
        out += p
    return out


def stage_events(events: list[tuple[str, int]], source: str, dest_split_dir: str, run_dir: str | None = None,
                 tar_name: str | None = None) -> int:
    """Extract the given events from packed tars into dest_split_dir (flat).
    source 'fs': the feature store tars <FS_DIR>/<sample>_<split>.tar (members event<ID>-graph.pyg.gz);
    source 'tar': one packed stage output <tar_name> (members event<ID>.pyg.gz)."""
    os.makedirs(dest_split_dir, exist_ok=True)
    want = {event_id(s, i) for s, i in events}
    tars = []
    if source == "fs":
        by_sample: dict[str, set[str]] = {}
        for s, i in events:
            by_sample.setdefault(s, set())
        for s in by_sample:
            for sp in ("train", "val", "test", "external"):
                p = os.path.join(FS_DIR, f"{s}_{sp}.tar")
                if os.path.exists(p):
                    tars.append(p)
    else:
        tars = [tar_name]
    n = 0
    for p in tars:
        with tarfile.open(p, "r") as tar:
            for m in tar:
                eid = m.name.split("event")[-1].split("-")[0].split(".")[0]
                if eid in want:
                    m.name = os.path.basename(m.name)
                    tar.extract(m, dest_split_dir, filter="data")
                    n += 1
    if n != len(want):
        raise RuntimeError(f"staged {n} of {len(want)} events into {dest_split_dir}")
    return n


def count_split_files(input_dir: str) -> list[int]:
    out = []
    for sp in ("trainset", "valset", "testset"):
        d = os.path.join(input_dir, sp)
        out.append(len([f for f in os.listdir(d) if f.endswith(".pyg.gz")]) if os.path.isdir(d) else 0)
    return out


# --------------------------------------------------------------------------------------------- metadata
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_metadata(extra: dict) -> dict:
    git = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()
    meta = {"git": git("rev-parse", "HEAD"), "git_dirty_code": git("status", "--porcelain", "--", "src", "env",
                                                                    "configs", "slurm"),
            "env_lock_sha256": _sha256(os.path.join(REPO, "env", "lock.txt")),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
            "start_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        import torch
        meta["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # pragma: no cover
        meta["gpu"] = None
    meta.update(extra)
    return meta


# --------------------------------------------------------------------------------------------- model classes
def model_class(stage: str, model: str):
    """acorn class by (stage, name), or one of our own classes by dotted path (e.g. hnl.models.triplet_gnn.TripletIGNN)."""
    if "." in model:
        import importlib
        mod, name = model.rsplit(".", 1)
        return getattr(importlib.import_module(mod), name)
    from acorn.core.core_utils import str_to_class
    return str_to_class(stage, model)


def checkpoint_class(ckpt: str):
    import torch
    hp = torch.load(ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]
    return model_class(hp["stage"], hp["model"])


# --------------------------------------------------------------------------------------------- training
class _Preempt:
    flag = False


class _LogMirror:
    """The CSVLogger writes to node-local disk and a daemon thread mirrors the version directory into
    stage_dir/lightning_logs/version_N (the layout read by hnl.train.curves). On HPC3's BeeGFS each
    metrics.csv append (every 100 steps) blocked the training loop for seconds to minutes: GPU idle ~50%.
    Copies go to a temporary name in the target directory and are then renamed; errors are printed,
    never raised (the mirror is a convenience copy, nothing downstream of training depends on it during the run).
    Syncs every `every_s` (60 s) and `delay_s` after each kick() (EpochLog kicks at every validation end), so a cancel
    without warning (free-gpu; the job's EXIT trap then removes the node-local copy) loses at most ~1 min of rows, and
    the epoch's validation row reaches /pub shortly after it is logged. Only whole lines are published: the copy is cut
    after its last newline, so a sync racing with a CSVLogger write never publishes a torn row."""

    def __init__(self, src: str, dst: str, every_s: float = 60.0, delay_s: float = 15.0):
        import threading
        self.src, self.dst, self.every_s, self.delay_s = src, dst, every_s, delay_s
        self._stop = threading.Event()
        self._kick = threading.Event()
        self._lock = threading.Lock()
        self._done: dict[str, int] = {}
        self._t = threading.Thread(target=self._loop, name="hnl-log-mirror", daemon=True)

    def start(self):
        self._t.start()
        return self

    def kick(self):
        self._kick.set()

    def sync(self):
        with self._lock:
            try:
                if not os.path.isdir(self.src):
                    return
                for name in sorted(os.listdir(self.src)):
                    s = os.path.join(self.src, name)
                    if not os.path.isfile(s):
                        continue
                    with open(s, "rb") as f:
                        data = f.read()
                    cut = data.rfind(b"\n") + 1      # whole lines only
                    data = data[:cut]
                    if not data or self._done.get(name) == len(data):
                        continue                    # nothing new: no /pub access
                    os.makedirs(self.dst, exist_ok=True)
                    tmp = os.path.join(self.dst, f".{name}.mirror_{os.getpid()}")
                    with open(tmp, "wb") as f:
                        f.write(data)
                    os.replace(tmp, os.path.join(self.dst, name))
                    self._done[name] = len(data)
            except Exception as e:  # noqa: BLE001
                print(f"[log-mirror] {self.src} -> {self.dst}: {e!r}", flush=True)

    def _loop(self):
        while not self._stop.is_set():
            if self._kick.wait(self.every_s):
                self._kick.clear()
                if self._stop.wait(self.delay_s):
                    break
            self.sync()

    def close(self):
        self._stop.set()
        self._kick.set()
        self._t.join(timeout=5)
        self.sync()


def _fail_fast_async_checkpoint_io():
    """Lightning's AsyncCheckpointIO (tensors cloned on the calling thread, one FIFO writer thread), except that once a
    background write has failed nothing later is submitted or written. With the stock plugin a failed best-*.ckpt write
    is followed by last/periodic/preempt.ckpt whose callback state names the missing file as best_model_path; the next
    chunk would resume from them and best_ckpt() would end at 'missing'. Here the failure surfaces at the next save (or
    at teardown) with no later checkpoint on disk, so the next chunk resumes from the last good one, as with synchronous
    writes (review of 2026-11-18)."""
    import torch
    from lightning_utilities.core.apply_func import apply_to_collection
    from pytorch_lightning.plugins.io import AsyncCheckpointIO

    def _clone(t):
        return t.detach().clone()

    class FailFastAsyncCheckpointIO(AsyncCheckpointIO):
        def save_checkpoint(self, *args, **kwargs):
            if self._error is not None:
                raise self._error
            self._ensure_setup()
            if "checkpoint" in kwargs:
                kwargs = {**kwargs, "checkpoint": apply_to_collection(kwargs["checkpoint"], torch.Tensor, _clone)}
            elif args:
                args = (apply_to_collection(args[0], torch.Tensor, _clone), *args[1:])

            def _save(*a, **k):
                if self._error is not None:          # an earlier queued write failed: drop this one
                    return
                for attempt in range(4):   # a BeeGFS blip (EREMOTEIO / ECOMM, 2026-11-19) is retried; the write is atomic
                    try:
                        self.checkpoint_io.save_checkpoint(*a, **k)
                        return
                    except OSError as ex:
                        if attempt == 3:
                            self._error = ex
                            return
                        print(f"[ckpt] write failed ({ex!r}), retry {attempt + 1}/3 in 30 s", flush=True)
                        time.sleep(30)
                    except BaseException as ex:  # noqa: BLE001
                        self._error = ex
                        return

            self._executor.submit(_save, *args, **kwargs)

    return FailFastAsyncCheckpointIO()


def _on_usr1(signum, frame):
    _Preempt.flag = True


def train(stage: str, arm: str, seed: int, run_dir: str, input_dir: str, max_epochs: int | None,
          overrides: dict | None = None, resume: str | None = None, limit_train_batches=None,
          config: str | None = None) -> dict:
    import torch
    from pytorch_lightning import Callback, Trainer, seed_everything
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from lightning_fabric.plugins.environments import LightningEnvironment
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from acorn.core.throughput import ThroughputCallback
    from acorn.stages.graph_construction.models import utils as gc_utils
    from acorn.utils.loading_utils import add_variable_name_prefix_in_config

    if stage == "ml" and not gc_utils.FRNN_AVAILABLE:
        raise RuntimeError("FRNN not available: refusing to train metric learning without it (acorn_pitfalls p03)")
    stage_dir = os.path.join(run_dir, stage)
    os.makedirs(stage_dir, exist_ok=True)
    cfg = yaml.safe_load(open(config or TEMPLATES[stage]))
    cfg.update({"input_dir": input_dir, "stage_dir": stage_dir, "data_split": count_split_files(input_dir)})
    if max_epochs is not None:
        cfg["max_epochs"] = max_epochs
    cfg.update(overrides or {})
    cfg_path = os.path.join(stage_dir, "train_config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    meta = run_metadata({"stage": stage, "arm": arm, "seed": seed, "config": cfg_path,
                         "config_sha256": _sha256(cfg_path), "data_split": cfg["data_split"], "resume": resume})

    t0 = time.time()
    seed_everything(seed, workers=True)
    pcfg = add_variable_name_prefix_in_config(copy.deepcopy(cfg))
    cls = model_class(pcfg["stage"], pcfg["model"])
    model = cls(pcfg)
    resume_cbs = []
    if stage == "gnn":
        # edge classifiers: acorn's own training loader -> the same epoch order, resumable mid-epoch (the Trunc*
        # classes of hnl.models.augment have it already), and Lightning's epoch counters repaired after a resume from
        # an epoch's last batch (hnl.train.resume_sampler)
        from hnl.train.resume_sampler import EpochCountersFix, use_resumable_train_loader
        if use_resumable_train_loader(model):
            print("[train] acorn training loader with hnl.train.resume_sampler (same order, resumable)", flush=True)
        resume_cbs.append(EpochCountersFix())
    meta["n_parameters"] = int(sum(p.numel() for p in model.parameters()))
    metric = cfg.get("metric_to_monitor", "val_loss")
    mode = cfg.get("metric_mode", "min")
    art = os.path.join(stage_dir, "artifacts")
    # save_top_k=-1: no best file is ever deleted. With 1, the validation that Lightning runs when a chunk stops at the
    # pre-timeout signal (after preempt.ckpt is written) could delete the epoch-end best that the next chunk restores as
    # best_model_path (arm (iii) of the training-sample study lost its epoch-28 checkpoint that way; DECISIONS
    # 2026-10-07 ~21:15 UTC). best_model_path / best_model_score are unchanged by this.
    best_cb = ModelCheckpoint(dirpath=art, filename="best-{epoch}-{" + metric + ":.6f}", monitor=metric, mode=mode,
                              save_top_k=-1, save_last=True, auto_insert_metric_name=True)
    # HNL_PERIODIC_MIN (default 30): minutes of training between periodic.ckpt writes (TrackML S4: 10, DECISIONS 2026-10-09)
    periodic_cb = ModelCheckpoint(dirpath=os.path.join(art, "periodic"), filename="periodic",
                                  train_time_interval=dt.timedelta(minutes=float(os.environ.get("HNL_PERIODIC_MIN", 30))),
                                  save_top_k=1, enable_version_counter=False)
    # every 5th epoch kept (weights of the run at equal training length, e.g. for shorter Stage-1 screening runs); only
    # after completed epochs, not when a chunk's pre-timeout stop ends one (hnl.train.resume_sampler)
    from hnl.train.resume_sampler import CompletedEpochCheckpoint
    epochs_cb = CompletedEpochCheckpoint(dirpath=os.path.join(art, "epochs"), filename="epoch{epoch}", every_n_epochs=5,
                                         save_top_k=-1, save_on_train_epoch_end=True, auto_insert_metric_name=False)

    class EpochLog(Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            m = {k: round(float(v), 6) for k, v in trainer.callback_metrics.items() if "/" not in k}
            print(f"[epoch] {trainer.current_epoch} step {trainer.global_step} t {time.time() - t0:.0f}s {m}",
                  flush=True)
            if mirror is not None:
                mirror.kick()   # publish the validation row soon (HNL_LOCAL_LOGS)

    # pre-timeout stop: preempt.ckpt at the next batch end, then stop without validating an unfinished epoch
    # (hnl.train.resume_sampler.PreemptStop; until 2026-10-08 Lightning validated at every stop)
    from hnl.train.resume_sampler import PreemptStop
    preempt_cb = PreemptStop(os.path.join(art, "preempt.ckpt"), lambda: _Preempt.flag)

    signal.signal(signal.SIGUSR1, _on_usr1)
    # a plain single-process environment: no dependence on SLURM_NTASKS / srun, no Lightning auto-requeue; the
    # pre-timeout signal is handled by preempt_cb above
    plugins = [LightningEnvironment()]
    # Both off unless set (slurm/baseline/_common.sh sets them by default, HNL_FASTIO=1): BeeGFS metadata calls from the
    # training loop stalled it for seconds to minutes (metrics.csv appends ~65%, checkpoint commits ~30% of the stalls)
    # - HNL_LOCAL_LOGS=<node-local dir>: CSVLogger there, mirrored to stage_dir/lightning_logs/version_N (_LogMirror)
    # - HNL_ASYNC_CKPT=1: checkpoints written by a background thread (Lightning's AsyncCheckpointIO clones the tensors
    #   on the calling thread first, so each file is the state at the moment of the save; fail-fast variant,
    #   _fail_fast_async_checkpoint_io); the same paths, so the ModelCheckpoint dirpath / best tracking and resume_ckpt
    #   are unchanged; the trainer's teardown waits for writes. The clones add the checkpoint's tensor size (a few MB
    #   here) to the recorded peak_gpu_mem_gib.
    local_logs = os.environ.get("HNL_LOCAL_LOGS")
    mirror = None
    if local_logs:
        version = CSVLogger(save_dir=stage_dir).version   # the next free version_N under stage_dir, as before
        # one local directory per stage_dir, so several trainings in one job never share a metrics.csv
        local_dir = os.path.join(local_logs, hashlib.sha1(os.path.abspath(stage_dir).encode()).hexdigest()[:12])
        logger = CSVLogger(save_dir=local_dir, version=version)
        mirror = _LogMirror(logger.log_dir, os.path.join(stage_dir, "lightning_logs", f"version_{version}")).start()
        print(f"[train] CSVLogger on {logger.log_dir}, mirrored to {mirror.dst}", flush=True)
    else:
        logger = CSVLogger(save_dir=stage_dir)
    if os.environ.get("HNL_ASYNC_CKPT") == "1":
        plugins.append(_fail_fast_async_checkpoint_io())
        print("[train] checkpoints: AsyncCheckpointIO (fail-fast)", flush=True)
    trainer = Trainer(accelerator="gpu", devices=1, num_nodes=1, max_epochs=cfg["max_epochs"],
                      callbacks=[best_cb, periodic_cb, epochs_cb, ThroughputCallback(), LearningRateMonitor("epoch"),
                                 preempt_cb, EpochLog(), *resume_cbs], enable_progress_bar=False,
                      logger=logger, precision=cfg.get("precision", 32),
                      strategy=DDPStrategy(find_unused_parameters=False, static_graph=True),
                      default_root_dir=stage_dir, plugins=plugins, limit_train_batches=limit_train_batches)
    torch.cuda.reset_peak_memory_stats()
    try:
        trainer.fit(model, ckpt_path=resume)
    finally:
        if mirror is not None:
            mirror.close()
    meta.update({"end_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "fit_seconds": round(time.time() - t0, 1),
                 "epochs_completed": int(trainer.current_epoch), "global_step": int(trainer.global_step),
                 "state": "preempted" if _Preempt.flag else "done",
                 "best_ckpt": best_cb.best_model_path, "best_score": float(best_cb.best_model_score)
                 if best_cb.best_model_score is not None else None,
                 "last_ckpt": best_cb.last_model_path,
                 "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)})
    with open(os.path.join(stage_dir, f"run_{os.environ.get('SLURM_JOB_ID', 'local')}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))
    return meta


# --------------------------------------------------------------------------------------------- inference
def infer(stage: str, ckpt: str, input_dir: str, out_dir: str, overrides: dict | None = None,
          accelerator: str = "gpu") -> dict:
    """accelerator "cpu" only for GNN scores that feed learned track building when no GPU slot is free (the same
    model and inputs; float32 numerics may differ from the GPU at the 1e-6 level)."""
    import torch
    from pytorch_lightning import Trainer
    from acorn.stages.graph_construction.models import utils as gc_utils
    from acorn.utils.loading_utils import add_variable_name_prefix_in_config

    if stage == "ml" and not gc_utils.FRNN_AVAILABLE:
        raise RuntimeError("FRNN not available: refusing to build graphs without it (acorn_pitfalls p03)")
    cls = checkpoint_class(ckpt)
    model = cls.load_from_checkpoint(ckpt, map_location=torch.device("cpu"))
    over = {"input_dir": input_dir, "stage_dir": out_dir, "data_split": count_split_files(input_dir),
            "hard_cuts": {}, "undirected": True, "skip_existing": False, "accelerator": accelerator, "devices": 1,
            "nodes": 1, "num_workers": [4, 4, 4]}
    if stage == "gnn":
        over["undirected_message_passing"] = True
    over.update(overrides or {})
    ck = model._hparams
    for k in ("node_features", "node_scales"):
        if k in over and list(over[k]) != list(ck.get(k, over[k])):
            raise ValueError(f"{k} differs from the checkpoint")
    model._hparams = add_variable_name_prefix_in_config({**ck, **over})
    model.setup(stage="predict")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    trainer = Trainer(accelerator=accelerator, devices=1, num_nodes=1, logger=False, default_root_dir=out_dir,
                      enable_progress_bar=False)
    with torch.inference_mode():
        trainer.predict(model)
    meta = run_metadata({"stage": stage, "ckpt": ckpt, "ckpt_sha256": _sha256(ckpt), "overrides": over,
                         "predict_seconds": round(time.time() - t0, 1)})
    with open(os.path.join(out_dir, f"infer_{os.environ.get('SLURM_JOB_ID', 'local')}.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return meta


def pack(src: str, dest: str) -> dict:
    """Pack every split directory of an acorn stage output into <dest>/<split>.tar (flat members)."""
    os.makedirs(dest, exist_ok=True)
    out = {}
    for sp in ("trainset", "valset", "testset"):
        d = os.path.join(src, sp)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".pyg.gz"))
        if not files:
            continue
        path = os.path.join(dest, f"{sp}.tar")
        with tarfile.open(path + ".tmp", "w") as tar:
            for f in files:
                tar.add(os.path.join(d, f), arcname=f)
        os.replace(path + ".tmp", path)
        out[sp] = len(files)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--arm", required=True)
    s.add_argument("--what", required=True, help="comma list of train,valmon,val,test,external")
    s.add_argument("--source", default="fs", choices=["fs", "tar"])
    s.add_argument("--tar-dir", default=None, help="for --source tar: directory holding <split>.tar files")
    s.add_argument("--dest", required=True)
    t = sub.add_parser("train")
    t.add_argument("--stage", required=True, choices=["ml", "gnn"])
    t.add_argument("--arm", required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--run-dir", required=True)
    t.add_argument("--input-dir", required=True)
    t.add_argument("--max-epochs", type=int, default=None)
    t.add_argument("--resume", default=None)
    t.add_argument("--limit-train-batches", type=float, default=None)
    t.add_argument("--set", action="append", default=[], help="key=yaml-value overrides")
    t.add_argument("--config", default=None, help="config template (default: configs/baseline/<stage>_train.yaml)")
    i = sub.add_parser("infer")
    i.add_argument("--stage", required=True, choices=["ml", "gnn"])
    i.add_argument("--ckpt", required=True)
    i.add_argument("--input-dir", required=True)
    i.add_argument("--out-dir", required=True)
    i.add_argument("--set", action="append", default=[])
    i.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"])
    p = sub.add_parser("pack")
    p.add_argument("--src", required=True)
    p.add_argument("--dest", required=True)
    a = ap.parse_args(argv)
    over = lambda kv: {x.split("=", 1)[0]: yaml.safe_load(x.split("=", 1)[1]) for x in kv}
    if a.cmd == "stage":
        for w in a.what.split(","):
            ev = arm_events(a.arm, w)
            if a.source == "fs":
                n = stage_events(ev, "fs", os.path.join(a.dest, ACORN_SPLIT[w]))
            else:
                n = stage_events(ev, "tar", os.path.join(a.dest, ACORN_SPLIT[w]),
                                 tar_name=os.path.join(a.tar_dir, f"{ACORN_SPLIT[w]}.tar"))
            print(f"[stage] arm {a.arm} {w}: {n} events -> {os.path.join(a.dest, ACORN_SPLIT[w])}", flush=True)
    elif a.cmd == "train":
        lim = a.limit_train_batches
        if lim is not None and lim >= 1:
            lim = int(lim)
        train(a.stage, a.arm, a.seed, a.run_dir, a.input_dir, a.max_epochs, over(a.set), a.resume, lim, a.config)
    elif a.cmd == "infer":
        print(json.dumps(infer(a.stage, a.ckpt, a.input_dir, a.out_dir, over(a.set), a.accelerator), indent=1,
                         default=str))
    elif a.cmd == "pack":
        print(json.dumps(pack(a.src, a.dest)))


if __name__ == "__main__":
    main()
