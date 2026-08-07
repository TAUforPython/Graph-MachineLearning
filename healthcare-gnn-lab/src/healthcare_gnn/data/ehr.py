"""Leakage guards and versioned phenotype helpers for MIMIC tables."""
from __future__ import annotations
import pandas as pd
HF_ICD10_PREFIXES=("I50",)
HF_ICD9_PREFIXES=("428",)

def is_heart_failure_code(code:str,version:int)->bool:
    normalized=str(code).replace(".","").upper()
    prefixes=HF_ICD10_PREFIXES if int(version)==10 else HF_ICD9_PREFIXES if int(version)==9 else ()
    return normalized.startswith(prefixes)

def history_before_index(events:pd.DataFrame,index_times:pd.Series)->pd.DataFrame:
    """Keep events strictly before each subject's externally defined index time."""
    required={"subject_id","event_time"}
    if not required<=set(events): raise ValueError(f"events require columns {sorted(required)}")
    mapped=events["subject_id"].map(index_times)
    if mapped.isna().any(): raise ValueError("Every subject needs an index time")
    return events.loc[events["event_time"]<mapped].copy()

def require_two_visits(admissions:pd.DataFrame)->pd.DataFrame:
    counts=admissions.groupby("subject_id")["hadm_id"].nunique()
    return admissions[admissions["subject_id"].isin(counts[counts>=2].index)].copy()
