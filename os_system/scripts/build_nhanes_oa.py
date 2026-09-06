"""Build the OA training CSV from public NHANES cycles (CDC, no NDA login)."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NHANES = ROOT / "data" / "nhanes"
OUT = ROOT / "data" / "osteoarthritis.csv"
BACKUP = ROOT / "data" / "osteoarthritis_matchthem.csv"

CYCLES = [("G", 2011), ("H", 2013), ("I", 2015), ("J", 2017)]


def read_xpt(name: str):
    path = NHANES / name
    if not path.exists():
        return None
    return pd.read_sas(path, format="xport")


def first_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def map_race(demo: pd.DataFrame) -> pd.Series:
    col = first_col(demo, ["RIDRETH3", "RIDRETH1"])
    raw = demo[col]
    if col == "RIDRETH3":
        return np.select([raw == 3, raw == 4, raw == 6], [1, 2, 3], default=0)
    return np.select([raw == 3, raw == 4], [1, 2], default=0)


def load_cycle(suf: str, year: int) -> pd.DataFrame:
    demo = read_xpt(f"DEMO_{suf}.xpt")
    bmx = read_xpt(f"BMX_{suf}.xpt")
    mcq = read_xpt(f"MCQ_{suf}.xpt")
    smq = read_xpt(f"SMQ_{suf}.xpt")
    diq = read_xpt(f"DIQ_{suf}.xpt")
    bpq = read_xpt(f"BPQ_{suf}.xpt")
    paq = read_xpt(f"PAQ_{suf}.xpt")
    osq = read_xpt(f"OSQ_{suf}.xpt")
    pfq = read_xpt(f"PFQ_{suf}.xpt")
    hsq = read_xpt(f"HSQ_{suf}.xpt")
    print(f"  merging {year}", flush=True)
    df = pd.DataFrame(
        {
            "SEQN": demo["SEQN"],
            "AGE": demo["RIDAGEYR"],
            "SEX": demo["RIAGENDR"],
            "RAC": map_race(demo),
        }
    )
    df = df.merge(bmx[["SEQN", "BMXBMI"]].rename(columns={"BMXBMI": "BMI"}), on="SEQN", how="left")
    typ = first_col(mcq, ["MCQ195", "MCQ191"])
    df = df.merge(
        mcq[["SEQN", "MCQ160A", typ]].rename(columns={"MCQ160A": "ARTH", typ: "TYPE"}),
        on="SEQN",
        how="left",
    )
    df = df.merge(smq[["SEQN", "SMQ020"]].rename(columns={"SMQ020": "SMK_RAW"}), on="SEQN", how="left")
    df = df.merge(diq[["SEQN", "DIQ010"]].rename(columns={"DIQ010": "DIA_RAW"}), on="SEQN", how="left")
    df = df.merge(bpq[["SEQN", "BPQ020"]].rename(columns={"BPQ020": "HTN_RAW"}), on="SEQN", how="left")
    activity = paq.copy()
    activity["PA"] = 0.0
    for col in ("PAQ650", "PAQ665", "PAQ605", "PAQ620"):
        if col in activity.columns:
            activity["PA"] = np.maximum(activity["PA"], (activity[col] == 1).astype(float))
    df = df.merge(activity[["SEQN", "PA"]], on="SEQN", how="left")
    if osq is not None and "OSQ060" in osq.columns:
        df = df.merge(osq[["SEQN", "OSQ060"]].rename(columns={"OSQ060": "OSP_RAW"}), on="SEQN", how="left")
    else:
        df["OSP_RAW"] = np.nan
    kn = first_col(pfq, ["PFQ061C"])
    walk = first_col(pfq, ["PFQ061A"])
    stair = first_col(pfq, ["PFQ061B"])
    pf_cols = ["SEQN"] + [c for c in (walk, stair, kn) if c]
    pf = pfq[pf_cols].copy()
    rename = {}
    if walk:
        rename[walk] = "WALK_RAW"
    if stair:
        rename[stair] = "STAIR_RAW"
    if kn:
        rename[kn] = "KNEE_RAW"
    df = df.merge(pf.rename(columns=rename), on="SEQN", how="left")
    for col in ("WALK_RAW", "STAIR_RAW", "KNEE_RAW"):
        if col not in df.columns:
            df[col] = np.nan
    if hsq is not None:
        gh = first_col(hsq, ["HSD010"])
        if gh:
            df = df.merge(hsq[["SEQN", gh]].rename(columns={gh: "GH_RAW"}), on="SEQN", how="left")
        else:
            df["GH_RAW"] = np.nan
    else:
        df["GH_RAW"] = np.nan
    df["YEAR"] = year
    return df


def build() -> pd.DataFrame:
    print("Reading NHANES XPT files...", flush=True)
    raw = pd.concat([load_cycle(suf, year) for suf, year in CYCLES], ignore_index=True)
    work = raw[(raw["AGE"] >= 40) & (raw["AGE"] <= 79)].copy()
    work["KOA"] = np.where(
        (work["ARTH"] == 1) & (work["TYPE"] == 2),
        1,
        np.where(work["ARTH"] == 2, 0, np.nan),
    )
    work = work.dropna(subset=["KOA", "BMI"]).copy()
    work["SMK"] = np.where(work["SMK_RAW"] == 1, 1, np.where(work["SMK_RAW"] == 2, 0, np.nan))
    work["DIA"] = np.where(work["DIA_RAW"] == 1, 1, np.where(work["DIA_RAW"].isin([2, 3]), 0, np.nan))
    work["HTN"] = np.where(work["HTN_RAW"] == 1, 1, np.where(work["HTN_RAW"] == 2, 0, np.nan))
    work["OSP"] = np.where(work["OSP_RAW"] == 1, 1, np.where(work["OSP_RAW"] == 2, 0, np.nan))
    work["KPN"] = np.where(work["KNEE_RAW"].isin([2, 3, 4]), 1, 0)
    work["MOB"] = (
        work["WALK_RAW"].isin([2, 3, 4])
        | work["STAIR_RAW"].isin([2, 3, 4])
        | work["KNEE_RAW"].isin([2, 3, 4])
    ).astype(int)
    gh = pd.to_numeric(work["GH_RAW"], errors="coerce")
    work["GH"] = gh.where(gh.between(1, 5))
    work["PA"] = work["PA"].fillna(0).clip(0, 1)
    work["SEX"] = work["SEX"].clip(1, 2)
    cols = ["AGE", "SEX", "BMI", "RAC", "SMK", "OSP", "DIA", "HTN", "PA", "KPN", "MOB", "GH", "KOA"]
    labelled = work[cols].copy()
    pos = labelled[labelled["KOA"] == 1]
    neg = labelled[labelled["KOA"] == 0].sample(n=len(pos), random_state=42)
    balanced = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"OA cases={len(pos)}  1:1 cohort={len(balanced)}  prevalence={balanced['KOA'].mean():.3f}", flush=True)
    return balanced


if __name__ == "__main__":
    if not BACKUP.exists() and (ROOT / "data" / "osteoarthritis.csv").exists():
        (ROOT / "data" / "osteoarthritis.csv").replace(BACKUP)
        print(f"Backed up MatchThem extract to {BACKUP.name}", flush=True)
    table = build()
    table.to_csv(OUT, index=False)
    print(f"Wrote {OUT}  shape={table.shape}", flush=True)
