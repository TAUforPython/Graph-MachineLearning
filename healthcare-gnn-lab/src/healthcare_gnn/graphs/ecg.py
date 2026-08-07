"""Twelve-lead ECG graph construction; no diagnosis labels enter edge creation."""
from __future__ import annotations
import numpy as np
LEADS=("I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6")
# Human-designed limb clique and precordial chain; this is not learned anatomy.
_PHYSIO=[(i,j) for group in (range(6),) for i in group for j in group if i<j]+[(i,i+1) for i in range(6,11)]

def physiological_adjacency() -> np.ndarray:
    A=np.zeros((12,12),dtype=np.float32)
    for i,j in _PHYSIO: A[i,j]=A[j,i]=1
    return A

def functional_adjacency(signal: np.ndarray, threshold: float) -> np.ndarray:
    """Absolute lead correlation for a single ``[12,time]`` record."""
    signal=np.asarray(signal,dtype=float)
    if signal.ndim!=2 or signal.shape[0]!=12: raise ValueError("signal must have shape [12, time]")
    A=np.nan_to_num(np.abs(np.corrcoef(signal)),nan=0.0); A[A<threshold]=0; np.fill_diagonal(A,0)
    return A.astype(np.float32)

def lead_adjacency(signal:np.ndarray|None=None,mode:str="physiological",threshold:float=0.5)->np.ndarray:
    if mode=="physiological": return physiological_adjacency()
    if signal is None: raise ValueError("functional and combined modes require a signal")
    functional=functional_adjacency(signal,threshold)
    if mode=="functional": return functional
    if mode=="combined": return np.maximum(physiological_adjacency(),functional)
    raise ValueError(f"Unknown lead graph mode: {mode}")
