.. MARS documentation master file, created by
   sphinx-quickstart on Wed Jul  7 11:59:47 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MARS's documentation!
================================
.. image:: img/mars_label.jpg
  :width: 80 %
  :align: center
  :target: https://github.com/quantumiracle/MARS

MARS is a ...

We also provide novices friendly `DRL Tutorials <https://github.com/tensorlayer/tensorlayer/tree/master/examples/reinforcement_learning>`_ for algorithms implementation, where each algorithm is implemented in an individual script. The tutorials serve as code examples for our Springer textbook `Deep Reinforcement Learning: Fundamentals, Research and Applications <https://deepreinforcementlearningbook.org/>`_ , you can get the free PDF if your institute has Springer license.

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   /introduction/intro

.. toctree::
   :maxdepth: 1
   :caption: Install MARS

   installation/install

.. toctree::
   :maxdepth: 3
   :caption: User Guide

   user_guide/quick_start
   user_guide/general
   user_guide/sarl
   user_guide/marl
   user_guide/notes

.. toctree::
   :maxdepth: 3
   :caption: MARS

   mars/rl/agent
   mars/rl/dqn
   mars/rl/ppo

   mars/marl/meta_learner

   mars/equilibrium_solver/solvers

   mars/utils/data_struct
   mars/utils/func
   mars/utils/logger

   mars/env/env

.. toctree::
   :maxdepth: 3
   :caption: Primal Experiments

   experiments/single_agent
   experiments/exploit

Contributing
==================

This project is under active development, if you want to join the core team, feel free to contact Zihan Ding at zhding[at]mail.ustc.edu.cn

Citation
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
