import json
from pathlib import Path
import pytest
from healthcare_gnn.training.run import run_smoke

@pytest.mark.parametrize("example",["mi","ptbxl","mimic","brain"])
def test_tiny_end_to_end(example,tmp_path:Path):
    config={"example":example,"mode":"synthetic","seed":4,"n_samples":50,"n_features":8,"k":2,"epochs":1,"models":["logistic","gcn","hyperbolic_gcn"],"strict_inductive":True}
    output=tmp_path/f"{example}.json"; result=run_smoke(config,output)
    assert result["status"]=="synthetic_smoke_not_clinical" and output.exists()
    assert len(json.loads(output.read_text())["models"])==3
