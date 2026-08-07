"""Small shared training runner used by every smoke notebook."""
from __future__ import annotations
import json, platform
from pathlib import Path
import numpy as np
import sklearn, torch, torch_geometric
from torch.nn import functional as F
from sklearn.inspection import permutation_importance
from healthcare_gnn.data.common import patient_split, seed_everything
from healthcare_gnn.data.synthetic import make_synthetic_cohort
from healthcare_gnn.graphs.build import build_graph
from healthcare_gnn.graphs.diagnostics import drop_edges, graph_diagnostics
from healthcare_gnn.models.baselines import make_baseline
from healthcare_gnn.models.gnn import HyperbolicGCN, NodeGNN
from healthcare_gnn.evaluation.metrics import bootstrap_auprc, classification_metrics, select_f1_threshold


def _versions() -> dict[str,str]:
    return {"python":platform.python_version(),"numpy":np.__version__,"sklearn":sklearn.__version__,"torch":torch.__version__,"torch_geometric":torch_geometric.__version__}

def _train_gnn(name:str,X:np.ndarray,y:np.ndarray,edge_index:np.ndarray,train:np.ndarray,val:np.ndarray,epochs:int,seed:int):
    x=torch.tensor(X); target=torch.tensor(y,dtype=torch.float32); edges=torch.tensor(edge_index,dtype=torch.long)
    train_nodes=torch.zeros(len(X),dtype=torch.bool); train_nodes[torch.tensor(train)]=True
    train_edges=edges[:,train_nodes[edges[0]] & train_nodes[edges[1]]]
    model=HyperbolicGCN(X.shape[1],32) if name=="hyperbolic_gcn" else NodeGNN(X.shape[1],32,name)
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-4); best=None
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad(); logits=model(x,train_edges); loss=F.binary_cross_entropy_with_logits(logits[train],target[train]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); optimizer.step()
        model.eval()
        with torch.no_grad(): score=F.binary_cross_entropy_with_logits(model(x,edges)[val],target[val]).item()
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        if best is None or score<best[0]: best=(score,state)
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad(): probability=torch.sigmoid(model(x,edges)).numpy()
    curvature=float(model.curvature.detach()) if isinstance(model,HyperbolicGCN) else None
    return probability,curvature

def run_smoke(config:dict[str,object],output:Path)->dict[str,object]:
    """Run a synthetic plumbing check; results are not clinical evidence."""
    seed=int(config["seed"]); seed_everything(seed)
    cohort=make_synthetic_cohort(int(config["n_samples"]),int(config["n_features"]),seed)
    split=patient_split(cohort.patient_ids,seed)
    graph=build_graph(cohort.X,method="knn",metric="gower",k=int(config["k"]),fit_indices=split.train,strict_inductive=bool(config["strict_inductive"]))
    records=[]
    for name in config["models"]:
        if name in {"logistic","random_forest","mlp"}:
            model=make_baseline(name,seed); model.fit(graph.transformed_X[split.train],cohort.y[split.train]); probability=model.predict_proba(graph.transformed_X)[:,1]
            importance=permutation_importance(model,graph.transformed_X[split.validation],cohort.y[split.validation],n_repeats=3,random_state=seed,scoring="average_precision").importances_mean.tolist()
            curvature=None
        else:
            probability,curvature=_train_gnn(name,graph.transformed_X,cohort.y,graph.edge_index,split.train,split.validation,int(config["epochs"]),seed); importance=None
        threshold=select_f1_threshold(cohort.y[split.validation],probability[split.validation])
        metrics=classification_metrics(cohort.y[split.test],probability[split.test],threshold)
        metrics["auprc_ci95"]=bootstrap_auprc(cohort.y[split.test],probability[split.test],cohort.patient_ids[split.test],seed)
        records.append({"model":name,"threshold_selected_on":"validation","threshold":threshold,"metrics":metrics,"curvature":curvature,"permutation_importance":importance})
    rewired=drop_edges(graph.edge_index,0.35,seed)
    payload={"schema_version":"1.0","status":"synthetic_smoke_not_clinical","example":config["example"],"seed":seed,"config_snapshot":config,"versions":_versions(),
      "split":{"unit":"patient","train":len(split.train),"validation":len(split.validation),"test":len(split.test)},
      "graph":graph_diagnostics(graph.edge_index,len(cohort.y),cohort.y),"edge_drop_audit":graph_diagnostics(rewired,len(cohort.y),cohort.y),"models":records,
      "prediction_schema":["deidentified_record_id","split","label","probability","seed"]}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    return payload
