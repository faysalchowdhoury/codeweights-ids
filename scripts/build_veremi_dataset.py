"""Parse the REAL VeReMi-Extension zips into a cached plausibility-feature table.

Input: the four attack-type zips from the VeReMi Extension (Zenodo 20090854),
each an outer zip of per-simulation inner zips of per-receiver trace JSON logs.

Labelling uses the REAL ground truth (`traceGroundTruthJSON`, type-4 records):
a received BSM (type-3) is an attack iff its reported position or speed deviates
from the true state for that (senderPseudo, messageID) beyond sensor noise, or
there is no ground-truth entry at all (a fabricated Sybil/ghost vehicle). The
attack *class* is the zip it came from.

Crucially, the IDS features are computed from the RECEIVED stream only (per-sender
trajectory self-consistency) -- never from the ground truth -- so they are exactly
what a deployed plausibility detector can see. `senderPseudo` / `messageID` are
kept as *leak* candidates (real simulation metadata that must not be used, because
it does not transfer across simulations).

Run isolated (the zip dir is untrusted input):
    py -3.12 -I scripts/build_veremi_dataset.py <raw_zip_dir> <out_csv> [--per-class N]

Output CSV (checked into data_sample/) lets the repo run fully offline with no
raw download. Columns: the 12 features + label + window + pseudo.
"""
import argparse
import io
import json
import math
import os
import zipfile
from collections import defaultdict

import numpy as np

FEATURE_NAMES = [
    "speed_mps", "accel_mag", "jerk_mag", "speed_resid", "pos_resid_m",
    "yaw_mag", "dt_anomaly", "replay_score", "ghost_score",
    "pseudo_norm", "msgid_norm", "heading_resid",
]
CLASS_NAMES = ["benign", "position_spoof", "vel_accel_fdi", "ghost", "replay_drift"]
ZIP_TO_CLASS = {                       # real VeReMi attack family -> our class
    "ConstPos_1416.zip": "position_spoof",
    "ConstSpeed_1416.zip": "vel_accel_fdi",
    "GridSybil_1416.zip": "ghost",
    "DataReplay_1416.zip": "replay_drift",
}
BSM_DT = 1.0            # real VeReMi BSM interval (measured: median 1.0 s)
POS_TOL = 10.0          # m ; GT position deviation above this => falsified
SPD_TOL = 5.0           # m/s ; GT speed deviation above this => falsified


def _ang(v):
    return math.atan2(v[1], v[0])


def _features(a, b, dt):
    pa, pb = np.array(a["pos"][:2]), np.array(b["pos"][:2])
    sa, sb = np.array(a["spd"][:2]), np.array(b["spd"][:2])
    ab = np.array(b.get("acl", [0, 0])[:2]); aa = np.array(a.get("acl", [0, 0])[:2])
    speed = float(np.linalg.norm(sb))
    accel = float(np.linalg.norm(ab))
    jerk = float(np.linalg.norm(ab - aa) / dt)
    vpos = float(np.linalg.norm(pb - pa) / dt)
    speed_resid = abs(speed - vpos)
    pos_resid = float(np.linalg.norm(pb - (pa + sa * dt)))
    hb, ha = b.get("hed", sb), a.get("hed", sa)
    yaw = abs(((_ang(hb) - _ang(ha) + math.pi) % (2 * math.pi)) - math.pi) / dt
    dt_anom = abs(dt - BSM_DT)
    replay = abs(float(b.get("rcvTime", 0)) - float(b.get("sendTime", 0)))
    ghost = max(0.0, pos_resid - 6.0) * (1.0 if speed < 2.0 else 0.0)
    head_resid = abs(((_ang(hb) - _ang(sb) + math.pi) % (2 * math.pi)) - math.pi) \
        if speed > 1.0 else 0.0
    return {
        "speed_mps": speed, "accel_mag": accel, "jerk_mag": jerk,
        "speed_resid": speed_resid, "pos_resid_m": pos_resid, "yaw_mag": yaw,
        "dt_anomaly": dt_anom, "replay_score": replay, "ghost_score": ghost,
        "pseudo_norm": (b["senderPseudo"] % 100000) / 100000.0,
        "msgid_norm": (b["messageID"] % 100000) / 100000.0,
        "heading_resid": head_resid,
    }


def _load_gt(iz):
    gt = {}
    for n in iz.namelist():
        if "GroundTruth" in n:
            for line in iz.read(n).decode("utf-8", "replace").splitlines():
                if line.strip():
                    o = json.loads(line)
                    gt[(o.get("senderPseudo"), o.get("messageID"))] = o
    return gt


def _is_attack(b, gt):
    g = gt.get((b["senderPseudo"], b["messageID"]))
    if g is None:
        return True                               # fabricated vehicle (Sybil/ghost)
    pd = np.linalg.norm(np.array(b["pos"][:2]) - np.array(g["pos"][:2]))
    sd = np.linalg.norm(np.array(b["spd"][:2]) - np.array(g["spd"][:2]))
    return bool(pd > POS_TOL or sd > SPD_TOL)


def process_zip(outer_path, cls, max_inner=2, max_files=500):
    rows = []
    z = zipfile.ZipFile(outer_path)
    inners = [n for n in z.namelist() if n.endswith(".zip")]
    for wi, name in enumerate(inners[:max_inner]):
        iz = zipfile.ZipFile(io.BytesIO(z.read(name)))
        gt = _load_gt(iz)
        traces = [n for n in iz.namelist() if "traceJSON" in n]
        unique = {}
        for tn in traces[:max_files]:
            for line in iz.read(tn).decode("utf-8", "replace").splitlines():
                if not line.strip():
                    continue
                o = json.loads(line)
                if o.get("type") == 3:
                    unique.setdefault((o["senderPseudo"], o["messageID"]), o)
        bysender = defaultdict(list)
        for o in unique.values():
            bysender[o["senderPseudo"]].append(o)
        win = f"{os.path.basename(outer_path)[:-4]}_w{wi}"
        for ps, msgs in bysender.items():
            msgs.sort(key=lambda m: m["sendTime"])
            for a, b in zip(msgs, msgs[1:]):
                dt = b["sendTime"] - a["sendTime"]
                if dt <= 0 or dt > 5:
                    continue
                label = CLASS_NAMES.index(cls) if _is_attack(b, gt) else 0
                feat = _features(a, b, dt)
                feat["label"] = label
                feat["window"] = win
                feat["pseudo"] = ps
                rows.append(feat)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("out_csv")
    ap.add_argument("--per-class", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import pandas as pd
    all_rows = []
    for zipname, cls in ZIP_TO_CLASS.items():
        p = os.path.join(args.raw_dir, zipname)
        if not os.path.exists(p):
            print(f"skip missing {zipname}"); continue
        r = process_zip(p, cls)
        print(f"{zipname:22s} -> {len(r):6d} pairs "
              f"({sum(x['label']>0 for x in r)} attack)")
        all_rows += r
    df = pd.DataFrame(all_rows)

    # balance: subsample benign (abundant) and each attack class
    rng = np.random.default_rng(args.seed)
    parts = []
    for c in range(len(CLASS_NAMES)):
        sub = df[df.label == c]
        k = min(args.per_class, len(sub))
        parts.append(sub.sample(k, random_state=args.seed) if len(sub) > k else sub)
    out = pd.concat(parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\nwrote {len(out)} rows -> {args.out_csv}")
    print("class balance:",
          {CLASS_NAMES[c]: int((out.label == c).sum()) for c in range(len(CLASS_NAMES))})
    print("windows:", sorted(out.window.unique()))


if __name__ == "__main__":
    main()
