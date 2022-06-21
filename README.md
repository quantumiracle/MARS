# MARS - Multi-Agent Research Studio
<img src="https://github.com/quantumiracle/MARS/blob/master/img/mars_label.jpg" alt="drawing" width="1000"/>


*If life exists on Mars, shall we human cooperate or compete with it?* 

## Status
WIP. Not released yet.

If you have any question (propose an ISSUE if it's general problem) or want to contribute to this repository, feel free to contact me: *zhding96@gmail.com*


## Usage Instruction

MARS is mainly built for solving **mult-agent Atari games** in [PettingZoo](https://www.pettingzoo.ml/atari).

A comprehensive usage [document](http://htmlpreview.github.io/?https://github.com/quantumiracle/MARS/blob/master/docs/build/html/index.html) is provided.

Some [tutorials](https://github.com/quantumiracle/MARS/tree/master/tutorial) are provided for simple MARL concepts, including building an arbitrary matrix game, solving the Nash equilibrium with different algorithms for matrix games, building arbitrary Markov game, solving Markov games, etc. 


MARS is still under-development and not prepared to release yet. You may find it hard to clone b.c. the author is testing algorithms with some models hosted on Git.



## Installation
Use Python 3.7 (Python 3.6 is not supported after Pettingzoo 1.12)
```
pip install -r requirements.txt
```

Run the following for installing [Atari ROM](https://github.com/Farama-Foundation/AutoROM) as a dependency of PettingZoo:
```
https://github.com/Farama-Foundation/AutoROM
```


## Quick Usage

### Basic:

**Train with MARL algorithm**:

```bash
python general_train.py --env pettingzoo_boxing_v1 --method nfsp --save_id train_0
```

**Exploit a trained model**:

```
python general_exploit.py --env pettingzoo_boxing_v1 --method nfsp --load_id train_0 --save_id exploit_0 --to_exploit second
```

More examples are provided in [`./examples/`](https://github.com/quantumiracle/MARS/tree/master/examples) and [`./unit_test/`](https://github.com/quantumiracle/MARS/tree/master/unit_test). Note that these files need to be put under the **root** directory (`./`) to run.

### Advanced:

**Train with MARL algorithm with multiprocess sampling and update**:

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

Basic RL Algorithms to do:
- [x] DQN
- [x] PPO
- [x] Genetic Algorithm
- [ ] PMOE
- [ ] DDPG
- [ ] TD3
- [ ] SAC

MARL Algorithms to do:
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
- [x] [PettingZoo]((https://www.pettingzoo.ml/atari))
- [x] [LaserTag](https://github.com/younggyoseo/lasertag-v0)
- [x] [SlimeVolley](https://github.com/hardmaru/slimevolleygym)
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

## Some Results

<img src="https://github.com/quantumiracle/MARS/blob/master/img/slimevolley-selfplay.gif" height=400 width=1000 >

Two agents in *SlimeVolley-v0* trained with self-play. 

<img src="https://github.com/quantumiracle/MARS/blob/master/img/boxing-selfplay.gif" height=500 width=400 >

Two agents in *Boxing-v1 PettingZoo* trained with self-play. 

[Exploitability](exploit.md) tests are also conducted.
