# Setup Guide

## Install Python Requirements
_**We strongly advise using a virtual envrionment manager such as Conda or pyenv**_

Install the environments and dependencies by running the following commands.
```
pip install -r requirements.txt
```

## Optional Setup
In this project, we use `robosuite` for visualization and sample-based motion planners. To use these functions, install `robosuite` by running the following commands.  
```
pip install robosuite==1.4.1
```
Then, set up the colors of collision meshes transparent in [these lines](https://github.com/ARISE-Initiative/robosuite/blob/eb01e1ffa46f1af0a3aa3ac363d5e63097a6cbcc/robosuite/utils/mjcf_utils.py#L18C39-L18C39) at `<robosuite-home>/utils/mjcf_utils.py`.

The simulated environment is built on [LEGATO](https://github.com/UT-HCRL/LEGATO), implemented by Mingyo Seo, and the Bi-RRT baseline in the main manuscript is based on [motion-planner](https://github.com/caelan/motion-planners), implemented by Caelan Reed Garrett. For using these, install them as submodules.
```
git submodule update --init --recursive
```


## Weights & Biases Logging
This repository uses [Weights & Biases](https://wandb.ai/) for logging and data monitoring during training. To setup,
sign up for a free account on the website and authorize your account using `wandb login`. This will require an API key
you can acquire from the [authorization page](https://wandb.ai/authorize). You should only need to do this setup step once.
