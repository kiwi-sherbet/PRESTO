# Downloading the Datasets and the Pre-Trained Models
Run the following command. By default, this downloads all relevant assets under `${REPO_ROOT}/data`.

```bash
python3 download.py
```


# Evaluating the Pre-Trained Models
To evaluate a trained model on the entire evaluation dataset, run the evaluation script as follows.
```bash
bash presto/scripts/eval_full.sh
```
Refer to [evaluate.py](presto/scripts/evaluate.py) regarding detailed script configuration.
To test a particular model, configure the `RUN_IDS` in [eval\_full.sh](presto/scripts/eval_full.sh) (default: run-326).


# Evaluating Bi-RRT (Optional)
To evaluate a trained model on the entire evaluation dataset, run the evaluation script as follows.
```bash
bash presto/scripts/eval_rrt.sh
```
Refer to [eval_rrt.py](presto/scripts/eval_rrt.py) regarding detailed script configuration.


# Training Your Own Model

You can train your own model as follows. The script by default asks for the name of your experiment interactively.
When the prompt `Name this run:` shows up, enter the experiment name and continue.

```bash
python3 presto/scripts/train.py model=kcfg train=trajopt data=kcfg
# Prompt:
# Name this run: [...]
```

To evaluate your own model, run the following commands.

```bash
python3 presto/scripts/parse_metric.py runs='[run-326]'
python3 presto/scripts/dump_metric.py runs='[run-326]'
# Output:
#  == Export `proc` to ./save/eval/proc_XXXX.pkl ==
```

Afterward, replace `./save/eval/proc_XXXX.pkl` in the appropriate `plot_*.py`.
Alternatively, a quick visualization is also available by adding `plot=1` argument to `dump_metric.py`.


# Visualization of Evaluation Results

The evaluation results can be visualized using the follow commands.
```bash
# for main result
python3 presto/scripts/plot_benchmark.py

# for ablation results
python3 presto/scripts/plot_ablation.py

# for guidance results
python3 presto/scripts/plot_guidance_ablation.py
```
To plot your own evaluation results, modify the script contents to reference the appropriate `proc.pkl` file.

# Troubleshooting

```bash
AttributeError: module 'OpenGL.EGL' has no attribute 'EGLDeviceEXT' 
# try: python3 -m pip install --upgrade PyOpenGL PyOpenGL_accelerate
```
