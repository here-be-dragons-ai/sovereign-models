"""Run moeve-lmbench on one (model, dataset) with all non-judge metrics of the task."""
import json, os, sys
model, dataset = sys.argv[1], sys.argv[2]
from lmbench.models import MODEL_MAPPING
from lmbench.dataset.dataset_utils import load_dataset
from lmbench.evaluation import EvaluationRunner
from lmbench.metrics import metrics_by_task
JUDGE = {"RagasQA", "RagasComparison", "RagasTopicExtraction", "Hallucination", "Values", "PoliticalParties", "WahlOMat"}
llm = MODEL_MAPPING["openai"].from_arg_string(
    model, "base_url=http://127.0.0.1:8100/v1,api_key=PROXY_KEY,ctx_size=4096,num_predict=512,total_parameters=7.45",
    temperature=0.0)
ds = load_dataset(dataset, False)
metrics = [m for m in metrics_by_task(ds.task) if not any(m.name.startswith(j) for j in JUDGE)]
print("metrics:", [m.name for m in metrics], flush=True)
EvaluationRunner(llm, ds).run(metrics=metrics, use_cache=False, use_metrics_cache=False)
