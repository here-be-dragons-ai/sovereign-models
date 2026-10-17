#!/bin/bash
cd /tmp/sm-lmbench/src
for d in german-quad-sample eurlexsum-german-sample; do
  for m in teuken-moeve teuken-correct; do
    out=/tmp/sm-lmbench/runs/$m/$d; mkdir -p $out
    echo "=== $m $d $(date +%T)"
    (cd $out && PROXY_KEY=x OPENAI_API_KEY= OUTPUT_FOLDER=$out CACHE_FOLDER=$out/cache MLFLOW_TRACKING_URI=file://$out/mlruns \
      /tmp/sm-lmbench/src/.venv/bin/python /tmp/sm-teuken-scripts/bench.py $m $d > $out/log.txt 2>&1; echo "exit $?" >> $out/log.txt)
    tail -1 $out/log.txt
  done
done
