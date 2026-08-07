"""Command-line entry point."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from healthcare_gnn.training.run import run_smoke

def main() -> None:
    parser=argparse.ArgumentParser(description="Leakage-aware healthcare GNN educational lab")
    parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output",type=Path,default=Path("reports/run.json"))
    args=parser.parse_args(); config=json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("mode")!="synthetic": raise SystemExit("This CLI increment runs only audited synthetic smoke mode. Use adapters and a reviewed full-data config for real data.")
    result=run_smoke(config,args.output); print(json.dumps({"output":str(args.output),"status":result["status"],"example":result["example"]}))
if __name__=="__main__": main()
