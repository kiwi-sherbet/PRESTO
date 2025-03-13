## Scripts

Directory:
* [download.py](download.py): Download training and evaluation datasets.
* [train.py](train.py): Train PRESTO.
* [evaluate.py](evaluate.py): Evaluate trained PRESTO.
* [eval\_full.sh](eval.sh): Thin wrapper around `evaluate.py` for iterating model evaluation across datasets.
* [parse\_metric.py](parse_metric.py): Collect evaluation logs into `.pkl` file.
* [dump\_metric.py](dump_metric.py): Aggregate metrics from `parse_metric.py` into stats for plotting.
