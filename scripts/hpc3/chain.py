"""Marker-driven job chains for HPC3 free-gpu / free (PreemptMode=CANCEL, kill_invalid_depend), 2026-11-17.

A step is a Slurm job script with arguments and a marker file that the script writes when the step is finished. Steps
are chained with --dependency=afterok on the jobs of the steps they need. On free-gpu a job can be cancelled by
preemption at any time; Slurm then also cancels every job that waits on it (kill_invalid_depend). `tick` repairs
that: every step whose marker is missing and that has no queued or running job is (re)submitted, with afterok on the
current jobs of its unfinished prerequisites. The training scripts resume from their latest checkpoint, so a
resubmission continues the run.

    python3 -B scripts/hpc3/chain.py add KEY --marker PATH [--after K1,K2] [--max-fail 3] -- [sbatch options] SCRIPT [ARGS]
    python3 -B scripts/hpc3/chain.py tick [--dry]       one pass (refuses while a watcher is alive, unless --force)
    python3 -B scripts/hpc3/chain.py watch [--poll 120]  tick in a loop (run it as a non-preemptible job:
                                                          slurm/hpc3/chain_watch.sbatch); stop with touch $DIR/STOP
    python3 -B scripts/hpc3/chain.py status             one line per step

State lives in $HNL_OUT/_chain (CHAIN_DIR overrides): defs/<KEY>.json (written by add, atomically) and
state/<KEY>.json (written by tick: job ids, states, attempts), log.txt, heartbeat. Job names are "c_<KEY>", so the
queue itself shows whether a step is in flight. Attempts: an attempt that ends without the marker in state PREEMPTED,
NODE_FAIL, BOOT_FAIL, TIMEOUT, or with exit code 75 (the scripts' "training not finished" code), or COMPLETED, counts as
interrupted (at most --max-interrupt, default 40); any other ending counts as a failure (at most --max-fail, default
3). A step that gives up blocks its dependents; `chain.py reset KEY` re-arms it.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

DIR = os.environ.get("CHAIN_DIR") or os.path.join(
    os.environ.get("HNL_OUT") or os.path.join(os.environ.get("HNL_DATA", "/pub/jinboz1/hnl_lrt"), "hpc3_v1"), "_chain")
DEFS, STATE = os.path.join(DIR, "defs"), os.path.join(DIR, "state")
INTERRUPTED = {"PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "TIMEOUT", "COMPLETED", "DEADLINE"}
HB_MAX_AGE = 900


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(os.path.join(DIR, "log.txt"), "a") as f:
        f.write(line + "\n")


def wjson(path: str, obj) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def defs() -> dict:
    out = {}
    if os.path.isdir(DEFS):
        for f in sorted(os.listdir(DEFS)):
            if f.endswith(".json"):
                d = json.load(open(os.path.join(DEFS, f)))
                out[d["key"]] = d
    return dict(sorted(out.items(), key=lambda kv: (kv[1].get("order", 0), kv[0])))


def state(key: str) -> dict:
    p = os.path.join(STATE, key + ".json")
    return json.load(open(p)) if os.path.exists(p) else {"key": key, "jobs": [], "interrupts": 0, "fails": 0}


QINFO: dict[str, tuple[str, str, str]] = {}   # job id -> (state, dependency, reason) of the last queued() call


def queued() -> dict[str, str]:
    """job name -> job id of this user's PENDING / RUNNING / etc. jobs (squeue); fills QINFO."""
    out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%j|%i|%T|%E|%r"],
                         capture_output=True, text=True, check=True).stdout
    q = {}
    QINFO.clear()
    for ln in out.splitlines():
        p = ln.split("|")
        if len(p) == 5 and p[2] not in ("COMPLETED", "CANCELLED", "FAILED", "PREEMPTED", "TIMEOUT", "NODE_FAIL"):
            q[p[0]] = p[1]
            QINFO[p[1]] = (p[2], p[3], p[4])
    return q


def final_state(jid: str) -> tuple[str, str]:
    out = subprocess.run(["sacct", "-n", "-X", "-P", "-j", jid, "-o", "State,ExitCode,Elapsed,NodeList"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        return "UNKNOWN", ""
    raw, code, el, node = (out[0].split("|") + ["", "", "", ""])[:4]
    return raw, f"exit {code} elapsed {el} node {node}"


def cmd_add(a) -> int:
    os.makedirs(DEFS, exist_ok=True)
    rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
    if not rest:
        sys.exit("add: give the sbatch options, script and arguments after the key")
    if any(x in ("-J", "--job-name") or x.startswith("--job-name=") for x in rest):
        sys.exit("add: no -J / --job-name in a step (the chain names its jobs c_<KEY>)")
    d = {"key": a.key, "marker": a.marker, "after": [k for k in (a.after or "").split(",") if k],
         "sbatch": rest, "max_fail": a.max_fail, "max_interrupt": a.max_interrupt, "order": a.order,
         "added": now(), "cwd": os.getcwd()}
    p = os.path.join(DEFS, a.key + ".json")
    if os.path.exists(p) and not a.replace:
        old = json.load(open(p))
        if {k: old.get(k) for k in ("marker", "after", "sbatch")} != {k: d[k] for k in ("marker", "after", "sbatch")}:
            sys.exit(f"add: {a.key} exists with a different definition (use --replace)")
        print(f"{a.key}: already defined")
        return 0
    wjson(p, d)
    print(f"{a.key}: defined")
    return 0


def heartbeat_alive() -> bool:
    p = os.path.join(DIR, "heartbeat")
    return os.path.exists(p) and time.time() - os.path.getmtime(p) < HB_MAX_AGE


def rewire(key: str, jid: str, d: dict, D: dict, q: dict, status: dict, dry: bool) -> bool | None:
    """A pending job whose afterok can never be satisfied (a prerequisite's job ended without its marker; Slurm's
    kill_invalid_depend has not cancelled it yet): point its dependency at the current jobs of its unfinished
    prerequisites. Returns True if rewired, None if a prerequisite has no job yet (try again next pass)."""
    deps = []
    for k in d["after"]:
        if k in D and os.path.exists(D[k]["marker"]):
            continue
        cur = q.get("c_" + k)
        if cur is None:
            return None
        deps.append(cur)
    spec = ("afterok:" + ":".join(deps)) if deps else ""
    if dry:
        log(f"{key}: DRY would rewire job {jid} to dependency '{spec}'")
        return True
    r = subprocess.run(["scontrol", "update", f"jobid={jid}", f"dependency={spec}"], capture_output=True, text=True)
    if r.returncode != 0:
        log(f"{key}: rewiring job {jid} to '{spec}' failed: {r.stderr.strip()}")
        return False
    log(f"{key}: job {jid} rewired to dependency '{spec or 'none'}'")
    return True


def tick(dry: bool = False) -> dict:
    os.makedirs(STATE, exist_ok=True)
    D = defs()
    q = queued()
    status = {}
    for key, d in D.items():
        s = state(key)
        name = "c_" + key
        if os.path.exists(d["marker"]):
            if s.get("status") != "done":
                s["status"] = "done"
                s["done_utc"] = now()
                log(f"{key}: done (marker {d['marker']})")
                if not dry:
                    wjson(os.path.join(STATE, key + ".json"), s)
            status[key] = "done"
            continue
        if name in q:
            jid = q[name]
            if jid not in s["jobs"]:  # submitted by hand or by an earlier tick that did not record it
                s["jobs"].append(jid)
                if not dry:
                    wjson(os.path.join(STATE, key + ".json"), s)
            st_q, dep, reason = QINFO.get(jid, ("", "", ""))
            if st_q == "PENDING" and ("(failed)" in dep or reason == "DependencyNeverSatisfied"):
                rewired = rewire(key, jid, d, D, q, status, dry)
                status[key] = ("rewired:" if rewired else "stuck:") + jid
                if rewired is not None:
                    status[key] = "active:" + jid if rewired else "stuck:" + jid
                continue
            status[key] = "active:" + jid
            continue
        # no job in the queue: account for the last one, then decide
        if s["jobs"] and s.get("accounted") != s["jobs"][-1]:
            jid = s["jobs"][-1]
            raw, info = final_state(jid)
            st = raw.split()[0] if raw else "UNKNOWN"
            if st in ("UNKNOWN", "PENDING", "RUNNING", "REQUEUED", "SUSPENDED"):
                status[key] = "waiting-sacct"
                continue
            if st == "CANCELLED" and raw.split()[-1] == str(os.getuid()):
                kind = "cancelled"  # scancel by this user: respected, the step stays stopped until `reset`
            elif st in INTERRUPTED or st == "CANCELLED" or info.startswith("exit 75:"):
                kind = "interrupted"  # preemption, time limit, dependency kill (kill_invalid_depend), unfinished training
            else:
                kind = "failed"
            if kind == "failed":
                s["fails"] += 1
            elif kind == "interrupted":
                s["interrupts"] += 1
            else:
                s["status"] = "cancelled"
            s["accounted"] = jid
            s.setdefault("history", []).append(f"{jid} {raw} {info} -> {kind}")
            log(f"{key}: job {jid} ended {raw} ({info}) without marker -> {kind} "
                f"(interrupts {s['interrupts']}, fails {s['fails']})")
            if not dry:
                wjson(os.path.join(STATE, key + ".json"), s)
        if s.get("status") == "cancelled":
            status[key] = "cancelled"
            continue
        if s["fails"] >= d["max_fail"] or s["interrupts"] >= d["max_interrupt"]:
            if s.get("status") != "gaveup":
                log(f"{key}: GIVING UP (fails {s['fails']}, interrupts {s['interrupts']}); chain.py reset {key} re-arms it")
                s["status"] = "gaveup"
                if not dry:
                    wjson(os.path.join(STATE, key + ".json"), s)
            status[key] = "gaveup"
            continue
        deps, blocked = [], None
        for k in d["after"]:
            if k not in D:
                blocked = f"unknown prerequisite {k}"
                break
            st_k = status.get(k)
            if st_k is None:  # defined later in the order: evaluate directly
                st_k = "done" if os.path.exists(D[k]["marker"]) else ("active:" + q["c_" + k] if "c_" + k in q else "idle")
            if st_k == "done":
                continue
            if st_k.startswith("active:"):
                deps.append(st_k.split(":", 1)[1])
            else:
                blocked = f"prerequisite {k} is {st_k}"
                break
        if blocked:
            status[key] = "blocked (" + blocked + ")"
            continue
        cmd = ["sbatch", "--parsable", "-J", name]
        if deps:
            cmd.append("--dependency=afterok:" + ":".join(deps))
        cmd += d["sbatch"]
        if dry:
            log(f"{key}: DRY would submit: {shlex.join(cmd)}")
            status[key] = "dry"
            continue
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=d.get("cwd") or None)
        jid = r.stdout.strip().split(";")[0]
        if r.returncode != 0 or not jid.isdigit():
            log(f"{key}: sbatch failed: {r.stderr.strip() or r.stdout.strip()}")
            status[key] = "submit-failed"
            continue
        s["jobs"].append(jid)
        s["status"] = "active"
        wjson(os.path.join(STATE, key + ".json"), s)
        log(f"{key}: submitted {jid}" + (f" after {','.join(deps)}" if deps else "") + f" (attempt {len(s['jobs'])})")
        status[key] = "active:" + jid
        q[name] = jid
    return status


def cmd_tick(a) -> int:
    if heartbeat_alive() and not a.force:
        sys.exit("tick: a watcher is alive (heartbeat younger than 15 min); use --force only if it is not")
    for k, v in tick(a.dry).items():
        print(f"{k}: {v}")
    return 0


def cmd_watch(a) -> int:
    os.makedirs(DIR, exist_ok=True)
    hb = os.path.join(DIR, "heartbeat")
    if heartbeat_alive() and open(hb).read().split()[:1] != [os.environ.get("SLURM_JOB_ID", "")] and not a.force:
        print("watch: another watcher is alive", file=sys.stderr)
        sys.exit(3)   # 3 = not started (chain_watch.sbatch does not restart on it)
    log(f"watcher start job {os.environ.get('SLURM_JOB_ID')} host {os.uname().nodename} poll {a.poll}s")
    while True:
        try:   # a BeeGFS error here (EREMOTEIO seen 2026-11-19) must not end the watcher
            with open(hb, "w") as f:
                f.write(f"{os.environ.get('SLURM_JOB_ID', '')} {os.uname().nodename} {os.getpid()} {now()}\n")
        except OSError as e:
            print(f"heartbeat write failed: {e!r}", flush=True)
        if os.path.exists(os.path.join(DIR, "STOP")):
            log("STOP found: watcher exits")
            os.remove(hb)
            return 0
        try:
            st = tick()
        except Exception as e:  # noqa: BLE001  (squeue hiccups etc.: try again next pass)
            try:   # the log is on BeeGFS too: an error while reporting an error must not end the watcher (2026-11-19)
                log(f"tick error: {e!r}")
            except Exception:  # noqa: BLE001
                pass
            st = {}
        if st and all(v in ("done", "gaveup", "cancelled") or v.startswith("blocked") for v in st.values()) \
                and not any(v.startswith("active") for v in st.values()) and a.exit_when_idle:
            log("nothing left to run: watcher exits")
            os.remove(hb)
            return 0
        time.sleep(a.poll)


def cmd_status(a) -> int:
    q = queued()
    for key, d in defs().items():
        s = state(key)
        cur = q.get("c_" + key)
        mark = "done" if os.path.exists(d["marker"]) else ("active " + cur if cur else s.get("status", "idle"))
        print(f"{key:40s} {mark:18s} jobs {','.join(s['jobs']) or '-':30s} int {s['interrupts']} fail {s['fails']}")
    return 0


def cmd_reset(a) -> int:
    s = state(a.key)
    s["fails"] = s["interrupts"] = 0
    s["status"] = "reset"
    wjson(os.path.join(STATE, a.key + ".json"), s)
    log(f"{a.key}: reset by hand")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("add")
    p.add_argument("key")
    p.add_argument("--marker", required=True)
    p.add_argument("--after", default="")
    p.add_argument("--max-fail", type=int, default=3)
    p.add_argument("--max-interrupt", type=int, default=40)
    p.add_argument("--order", type=int, default=0, help="submission priority (lower first), then key")
    p.add_argument("--replace", action="store_true")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p = sp.add_parser("tick")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--force", action="store_true")
    p = sp.add_parser("watch")
    p.add_argument("--poll", type=int, default=120)
    p.add_argument("--force", action="store_true")
    p.add_argument("--exit-when-idle", action="store_true")
    sp.add_parser("status")
    p = sp.add_parser("reset")
    p.add_argument("key")
    a = ap.parse_args(argv)
    os.makedirs(DIR, exist_ok=True)
    return {"add": cmd_add, "tick": cmd_tick, "watch": cmd_watch, "status": cmd_status, "reset": cmd_reset}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
