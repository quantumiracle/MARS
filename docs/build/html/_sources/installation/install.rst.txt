Installation
=================================

MARS requires PyTorch.

To use all supported environments in MARS, you need to install those environments, including `OpenAI Gym <https://gym.openai.com/>`_, `PettingZoo <https://github.com/PettingZoo-Team/PettingZoo>`_,
`SlimeVolley <https://github.com/hardmaru/slimevolleygym>`_, `LaserTag <https://github.com/younggyoseo/lasertag-v0>`_, etc.

Direct installation: 

.. code-block:: bash
   :linenos:
   conda create -n mars python==3.6 -y
   conda activate mars
   pip3 install -r requirements
   pip3 install mars --upgrade

Install from the source code on github:

.. code-block:: bash
   :linenos:

   git clone https://github.com/quantumiracle/MARS.git
   cd MARS
   pip3 install .
   pip3 install -r requirements

.. Note:: Direct installation through pip is not developed yet.


.. Tip:: Better to use Python 3.7 instead of 3.6.

.. WARNING:: Be careful with the verison of package PettingZoo.


