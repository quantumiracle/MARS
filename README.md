# MARS - Multi-Agent Research Studio
<img src="https://github.com/quantumiracle/MARS/blob/master/img/mars_label.jpg" alt="drawing" width="1000"/>


*If life exists on Mars, shall we human cooperate or compete with it?* 

## Usage Instruction
See [here](http://htmlpreview.github.io/?https://github.com/quantumiracle/MARS/blob/master/docs/build/html/index.html)


## Development
Basic RL Algorithms to do:
- [x] DQN
- [x] PPO
- [x] Genetic Algorithm
- [ ] PMOE
- [ ] DDPG

MARL Algorithms to do:
- [x] Self-Play
- [ ] [Fictitious Self-Play](http://proceedings.mlr.press/v37/heinrich15.pdf)
- [x] Neural Fictitious Self-Play
- [ ] Policy Space Responce Oracle
- [ ] [Joint Policy Space Responce Oracle](http://proceedings.mlr.press/v139/marris21a.html)
- [ ] [MADDPG](https://arxiv.org/abs/1706.02275)
- [ ] QMIX
- [ ] QTRAN
- [ ] MAPPO
- [x] Nash-DQN
- [x] Nash-DQN-Exploiter

Supported environments:
- [x] Openai Gym
- [x] PettingZoo
- [x] LaserTag
- [x] SlimeVolley
- [ ] SMAC

## Self-play
<img src="https://github.com/quantumiracle/MARS/blob/master/img/slimevolley-selfplay.gif" height=400 width=1000 >

Two agents in *SlimeVolley-v0* trained with self-play. 

<img src="https://github.com/quantumiracle/MARS/blob/master/img/boxing-selfplay.gif" height=500 width=400 >

Two agents in *Boxing-v1 PettingZoo* trained with self-play. 

[Exploitability](exploit.md) tests are also conducted.
