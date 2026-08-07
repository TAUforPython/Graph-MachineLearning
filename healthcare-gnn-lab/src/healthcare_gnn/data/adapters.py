"""Public/credentialed dataset adapters with explicit access boundaries."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import pandas as pd

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    access: str
    license_note: str

SPECS = {
    "uci_mi": DatasetSpec("UCI MI Complications", "https://archive.ics.uci.edu/dataset/579/myocardial+infarction+complications", "public; inspect source terms", "Dataset-specific terms apply."),
    "ptbxl": DatasetSpec("PTB-XL", "https://physionet.org/content/ptb-xl/", "public under PhysioNet terms", "PhysioNet credential/use requirements apply."),
    "mimic_demo": DatasetSpec("MIMIC-IV Demo", "https://physionet.org/content/mimic-iv-demo/", "open demo", "Demo is for pipeline checks, not performance claims."),
    "mimic_full": DatasetSpec("MIMIC-IV", "https://physionet.org/content/mimiciv/", "credentialed; never auto-download", "Complete PhysioNet credentialing and DUA."),
    "abide": DatasetSpec("ABIDE I/II", "https://fcon_1000.projects.nitrc.org/indi/abide/", "public subject to source terms", "Site and derivative terms apply."),
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path: Path, source: str, files: list[Path]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source": source, "files": [{"path": str(f), "sha256": sha256(f)} for f in files]}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_uci_mi_csv(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load an already obtained UCI table; target choice must follow a prevalence report."""
    frame = pd.read_csv(path)
    if target not in frame:
        raise ValueError(f"Unknown target {target!r}; available columns: {list(frame)}")
    y = frame[target]
    if y.dropna().nunique() != 2:
        raise ValueError("The configured smoke task requires a binary target")
    return frame.drop(columns=[target]), y


def require_local_mimic(root: Path) -> Path:
    """Refuse automatic access to credentialed MIMIC-IV."""
    if not root.exists():
        raise FileNotFoundError("MIMIC files are absent. Download them manually after PhysioNet approval; automatic credentialed downloads are disabled.")
    return root


def choose_mimic_demo_target(next_hf_positives: int, minimum: int = 10) -> str:
    """Make the documented demo fallback explicit instead of fabricating HF labels."""
    return "next_visit_heart_failure" if next_hf_positives >= minimum else "early_mortality_tutorial"
