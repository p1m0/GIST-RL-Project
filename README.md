# GIST-RL-Project

We support three MuJoCo environments (but there is no reason other's shouldn't work):
* Humanoid-v5
* HalfCheetah-v5
* Ant-v5

*IMPORTANT!* <br>
Run all scripts from the root directory using "PYTHONPATH=$PYTHONPATH:$(pwd) python scripts/run_XXX.py"
Install required packages from requirements.txt using "pip install -m requirements.txt"

Some scripts require some arguments (like env or experiment name). This should get unified later to make it simpler to use. We could also create some predefined configs to not have to type out long arguments.

The run_bc.py script downloads expert data from Minari (https://minari.farama.org/main/) and trains for some number of epochs on them. There is no DAgger implemented as of yet. For that we need to get an expert policy (probably by training a more advanced RL model from stable baselines).

The run_a2c.py script runs the Advantage Actor Critic (Policy gradient with a few tricks) algorithm on a given MuJoCo environment. There is also Generalized Advantage Estimation (GAE) available for use (another fancy trick which might improve performance a little bit)

The run_sac.py runs the Soft Actor Critic algorithm on a given MuJoCo environment. This is a fancier version of DeepQ that works with continuous action spaces. It uses reparametrization trick, Polyak averaging and some other tricks to improve performance (I'm not gonna act like I understand 100% how and why they work).