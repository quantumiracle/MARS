# Some Notes
* For SlimeVolley environments, if you want to train single agent against the baseline provided by the environment, you need to: 
  1. set *against_baseline* as *True* in either the yaml file or input arguments;
  2. instantiate two agent models in `run.py` and fix the first one, so that only the second model is learnable since in single-agent SlimeVolley environments the learnable one is set as the second one by default.