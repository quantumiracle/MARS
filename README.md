# MARS - Multi-Agent Research Studio
<img src="https://github.com/quantumiracle/MARS/blob/master/img/mars_label.jpg" alt="drawing" width="1000"/>


*If life exists on Mars, shall we human cooperate or compete with it?* 

**Table of contents:**

- [Status](#status)
- [Installation](#installation)
- [Usage](#usage)
  - [Description](#description)
  - [Support](#support)
  - [Quick Start](#quick-start)
  - [Advanced Usage](#advanced-usage)
- [Development](#development)
- [License](#license)
- [Citation](#citation)
- [Primary Results](#primary-results)

## Status
WIP. Not released yet.

If you have any question (propose an ISSUE if it's general problem) or want to contribute to this repository, feel free to contact me: *zhding96@gmail.com*


## Installation
Use Python 3.7 (Python 3.6 is not supported after Pettingzoo 1.12)
```
pip install -r requirements.txt
```

Run the following for installing [Atari ROM](https://github.com/Farama-Foundation/AutoROM) as a dependency of PettingZoo:
```
https://github.com/Farama-Foundation/AutoROM
```


## Usage

### Description

MARS is mainly built for solving **mult-agent Atari games** in [PettingZoo](https://www.pettingzoo.ml/atari), especially competitive (zero-sum) games.

A comprehensive usage [document](http://htmlpreview.github.io/?https://github.com/quantumiracle/MARS/blob/master/docs/build/html/index.html) is provided.

Some [tutorials](https://github.com/quantumiracle/MARS/tree/master/tutorial) are provided for simple MARL concepts, including building an arbitrary matrix game, solving the Nash equilibrium with different algorithms for matrix games, building arbitrary Markov game, solving Markov games, etc. 


MARS is still under-development and not prepared to release yet. You may find it hard to clone b.c. the author is testing algorithms with some models hosted on Git.

### Support
The `EnvSpec = Environment type + '_' + Environment Name` as a convention in MARS.

Supported environments are as following:

| Environment Type      | Environment Name |
| --------------- | --------------------------------------------------|
| [`gym`](https://github.com/openai/gym) | all standard envs in OpenAI Gym |
|[`pettingzoo`](https://www.pettingzoo.ml) | 'basketball_pong_v3', 'boxing_v2', 'combat_jet_v1', 'combat_tank_v2', 'double_dunk_v3', 'entombed_competitive_v3', 'entombed_cooperative_v3', 'flag_capture_v2', 'foozpong_v3', 'ice_hockey_v2', 'joust_v3','mario_bros_v3', 'maze_craze_v3', 'othello_v3', 'pong_v3', 'quadrapong_v4', 'space_invaders_v2', 'space_war_v2', 'surround_v2', 'tennis_v3', 'video_checkers_v4', 'volleyball_pong_v2', 'warlords_v3', 'wizard_of_wor_v3';  'dou_dizhu_v4', 'go_v5', 'leduc_holdem_v4', 'rps_v2', 'texas_holdem_no_limit_v6', 'texas_holdem_v4', 'tictactoe_v3', 'uno_v4' |
|[`lasertag`](https://github.com/younggyoseo/lasertag-v0) | 'LaserTag-small2-v0', 'LaserTag-small3-v0', 'LaserTag-small4-v0' |
|[`slimevolley`](https://github.com/hardmaru/slimevolleygym) | 'SlimeVolley-v0', 'SlimeVolleySurvivalNoFrameskip-v0', 'SlimeVolleyNoFrameskip-v0', 'SlimeVolleyPixel-v0' | 
|[`robosumo`](https://github.com/openai/robosumo) | 'ant_vs_ant_v0', etc |
|[`mdp`](https://github.com/quantumiracle/MARS/tree/master/mars/env/mdp)| 'arbitrary_mdp', 'arbitrary_richobs_mdp', 'attack', 'combinatorial_lock' |

Supported algorithms are as following:
| Method      |
| --------------- |
| Self-play |
| Fictitious Self-play |
| Neural Fictitious Self-play |
| Policy Space Response Oracle |
| Nash Q-learning |
| Nash Value Iteration |
| Nash DQN |
| Nash DQN with Exploiter |

### Quick Start:

**Train with MARL algorithm**:

`python general_train.py --env **EnvSpec** --method **Method** --save_id **WheretoSave**`

```bash
# PettingZoo Boxing_v1, neural fictitious self-play
python general_train.py --env pettingzoo_boxing_v1 --method nfsp --save_id train_0

# PettingZoo Pong_v2, fictitious self-play
python general_train.py --env pettingzoo_pong_v2 --method fictitious_selfplay --save_id train_1

# PettingZoo Surround_v1, policy space response oracle
python general_train.py --env pettingzoo_surround_v1 --method prso --save_id train_3

# SlimeVolley SlimeVolley-v0, self-play
python general_train.py --env slimevolley_SlimeVolley-v0 --method selfplay --save_id train_4
```

**Exploit a trained model**:

`python general_exploit.py --env **EnvSpec** --method **Method** --load_id **TrainedModelID** --save_id **WheretoSave** --to_exploit **ExploitWhichPlayer**`

```
python general_exploit.py --env pettingzoo_boxing_v1 --method nfsp --load_id train_0 --save_id exploit_0 --to_exploit second
```

More examples are provided in [`./examples/`](https://github.com/quantumiracle/MARS/tree/master/examples) and [`./unit_test/`](https://github.com/quantumiracle/MARS/tree/master/unit_test). Note that these files need to be put under the **root** directory (`./`) to run.

### Advanced Usage:

**Train with MARL algorithm with multiprocess sampling and update**:

`python general_launch.py --env **EnvSpec** --method **Method** --save_id **WheretoSave**`

```
python general_launch.py --env pettingzoo_boxing_v1 --method nfsp --save_id multiprocess_train_0
```

**Exploit a trained model (same as above)**:

```
python general_exploit.py --env pettingzoo_boxing_v1 --method nfsp --load_id multiprocess_train_0 --save_id exploit_0 --to_exploit second
```

**Test a trained MARL model in single-agent Atari**:

This function is for limited environments (like *boxing*) since not all envs in PettingZoo Atari has a single-agent counterpart in OpenAI Gym.

```
python general_test.py --env pettingzoo_boxing_v1 --method nfsp --load_id train_0 --save_id test_0
```

**Bash script for server**:

Those bash scripts to run multiple tasks on servers are provided in `./server_bash_scripts`. For example, to run a training bash script (put it in the **root** directory):

```bash
./general_train.sh
```



## Development

Basic single-agent RL algorithms (for best response, etc) to do:
- [x] DQN
- [x] PPO
- [x] Genetic Algorithm
- [ ] PMOE
- [ ] DDPG
- [ ] TD3
- [ ] SAC

MARL algorithms to do:
- [x] Self-Play
- [x] [Fictitious Self-Play](http://proceedings.mlr.press/v37/heinrich15.pdf)
- [x] [Neural Fictitious Self-Play](https://arxiv.org/abs/1603.01121)
- [x] [Policy Space Responce Oracle](https://proceedings.neurips.cc/paper/2017/file/3323fe11e9595c09af38fe67567a9394-Paper.pdf)
- [ ] [Joint Policy Space Responce Oracle](http://proceedings.mlr.press/v139/marris21a.html)
- [ ] [MADDPG](https://arxiv.org/abs/1706.02275)
- [ ] QMIX
- [ ] QTRAN
- [ ] MAPPO
<!-- - [x] Nash-DQN
- [x] Nash-DQN-Exploiter
 -->
  Supported environments:
- [x] [Openai Gym](https://github.com/openai/gym)
- [x] [PettingZoo](https://www.pettingzoo.ml)
- [x] [LaserTag](https://github.com/younggyoseo/lasertag-v0)
- [x] [SlimeVolley](https://github.com/hardmaru/slimevolleygym)
- [x] [Robosumo](https://github.com/openai/robosumo)
- [x] [Matrix Markov Game](https://github.com/quantumiracle/MARS/tree/master/mars/env/mdp)
- [ ] SMAC

## License

MARS is distributed under the terms of Apache License (Version 2.0).

See [Apache License](https://github.com/quantumiracle/MARS/blob/master/LICENSE) for details.

## Citation

If you find MARS useful, please cite it in your publications.

```
@software{MARS,
  author = {Zihan Ding},
  title = {MARS},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/quantumiracle/MARS}},
}
```

## Primary Results

<img src="https://github.com/quantumiracle/MARS/blob/master/img/slimevolley-selfplay.gif" height=400 width=1000 >

Two agents in *SlimeVolley-v0* trained with self-play. 

<img src="https://github.com/quantumiracle/MARS/blob/master/img/boxing-selfplay.gif" height=500 width=400 >

Two agents in *Boxing-v1 PettingZoo* trained with self-play. 

[Exploitability](exploit.md) tests are also conducted.
