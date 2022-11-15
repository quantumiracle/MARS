# MARL Backgrounds

## Cooperative Games

| Algorithm     | Critic                                                       | Actor             | Training | Note                                                         |
| ------------- | ------------------------------------------------------------ | ----------------- | -------- | ------------------------------------------------------------ |
| VDN           | $Q(\mathbf{o}, \mathbf{u})$                                  |                   |          | value decomposition                                          |
| QMIX          |                                                              |                   |          |                                                              |
| Weighted-QMIX |                                                              |                   |          |                                                              |
| QTRAN         |                                                              |                   |          |                                                              |
| QAtten        |                                                              |                   |          |                                                              |
| MAVEN         |                                                              |                   |          |                                                              |
| MATD3         |                                                              |                   |          |                                                              |
| CoMIX         |                                                              |                   |          |                                                              |
| FacMADDPG     |                                                              |                   |          |                                                              |
| COMA          | $Q(\mathbf{x}, \mathbf{u})$                                  | $\pi_i(u_i\|o_i)$ | CTDE     | counterfactual baseline                                      |
| MADDPG        | $Q(\mathbf{x}, \mathbf{u})$                                  | $\mu_i(u_i\|o_i)$ | CTDE/DT  | approximate policies of other agents as $\hat{\mu}^j_i(o_j)$ |
| MAAC          | $Q(\mathbf{o}, \mathbf{u})$                                  | $\pi_i(u_i\|o_i)$ | CTDE     | centralized critics share an attention mechanism             |
| IPPO          | $V_i(o_i)$                                                   | $\pi_i(u_i\|o_i)$ | DT       | use the team reward $r(\mathbf{x}, \mathbf{u})$ to approximate $r(o_i , u_i )$ |
| MAPPO         | $V(\mathbf{x} \text{ or } \mathbf{o})$ (actually $V(\mathbf{x}_i$)) | $\pi_i(u_i\|o_i)$ | CTDE     | the critic and actor can be shared across agents; the global state $\mathbf{x}$ can be agent-specific $\mathbf{x}_i$ |
| MATRPO        | $V_i(o_i)$ ($A_i(o_i, \mathbf{u})$)                          | $\pi_i(u_i\|o_i)$ | DT       | share local likelihood ratio $\Upsilon_n(\theta_i)=\frac{\pi_i(u_n|o_i;\theta_i)}{\pi_i(u_n|o_i;\theta_{old})}$ with neighbors |
| MADDPG        |                                                              |                   | DT       |                                                              |
| HAPPO         |                                                              |                   |          |                                                              |
|               |                                                              |                   |          |                                                              |

*Notes*:

- CTDE: centralized training, decentralized execution

- DT: decentralized training

- $\mathbf{x}$: some underlying true state of the environment; 

- $\mathbf{o}$: concatenation of observations from all agents;

- $\mathbf{u}$: vector of actions by all agents;

- $\mu_i$: deterministic policy of agent $i$;

- $\pi_i$: stochastic policy of agent $i$; 

- $\mathbf{x}_i$: an agent $i$ specific global state used in MAPPO; 

## Competitive Games

### Matrix Game

| Algorithm          |      |      |
| ------------------ | ---- | ---- |
| GDA                |      |      |
| OGDA               |      |      |
| EG                 |      |      |
| OMWU               |      |      |
| GDmax              |      |      |
| MGDA               |      |      |
| two time-scale GDA |      |      |
| SGDA               |      |      |
| Self-Play          |      |      |
| Fictitious Play    |      |      |
| Double Oracle      |      |      |



### Extensive-form Game

| Algorithm |      |      |
| --------- | ---- | ---- |
| NFSP      |      |      |
| PSRO      |      |      |
|           |      |      |



### Markov Game

| Algorithm               |      |      |
| ----------------------- | ---- | ---- |
| Nash Q-learning         |      |      |
| Nash Value Iteration    |      |      |
| Nash V-learning         |      |      |
| Nash DQN                |      |      |
| Nash DQN with Exploiter |      |      |

