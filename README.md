# GIST-RL-Project

We support three MuJoCo environments (but there is no reason other's shouldn't work):
* Humanoid-v5
* HalfCheetah-v5
* Ant-v5

This is a nice comparison of the model performance: https://tianshou.org/en/v0.4.8/tutorials/benchmark.html
The problem is that the models there reach a much better performance than what I was able to do. Also according to their benchmarks A2C should be much better than what we get. Idk why.
Here is a decent explanation why HalfCheetah performs so poorly: https://www.perplexity.ai/search/is-there-any-elegant-way-to-fi-wsDg3Tp5Sga22xrBU3tWjg

*IMPORTANT!* <br>
Run all scripts from the root directory using "PYTHONPATH=$PYTHONPATH:$(pwd) python scripts/run_XXX.py"
Install required packages from requirements.txt using "pip install -m requirements.txt"
To see graphs with training process and statistics run "tensorboard --logdir=data" from the root folder and open the link that shows up.

Some scripts require some arguments (like env or experiment name). This should get unified later to make it simpler to use. We could also create some predefined configs to not have to type out long arguments.

The run_bc.py script downloads expert data from Minari (https://minari.farama.org/main/) and trains for some number of epochs on them. There is no DAgger implemented as of yet. For that we need to get an expert policy (probably by training a more advanced RL model from stable baselines).

The run_a2c.py script runs the Advantage Actor Critic (Policy gradient with a few tricks) algorithm on a given MuJoCo environment. There is also Generalized Advantage Estimation (GAE) available for use (another fancy trick which might improve performance a little bit)

The run_sac.py runs the Soft Actor Critic algorithm on a given MuJoCo environment. This is a fancier version of DeepQ that works with continuous action spaces. It uses the reparametrization trick, entropy, Polyak averaging and some other tricks to improve performance (I'm not gonna act like I understand 100% how and why they work).

*FINDINGS* <br>
As expected, the A2C is much less sample efficient than SAC. This is because A2C is an on-policy method meaning for every training steps it needs to get new transitions from the environment. SAC on the other hand is much more computationally expensive but usually reaches better performance than A2C ever could. There are some differences between our implementation and SB3's implementation, the main being stability - our training curves are much more spikey and sometimes the evaluation performance crashes while the SB3 runs go very smoothly. The most obvious reason is probably a difference in implementation. Another difference might be different replay buffer (if they use a prioritized replay buffer that might help increase performance and smoothen out the training process).

Behavioral cloning seems like it can work SOMETIMES. For example with the Ant environment it reached much better performance than SAC or A2C with "only" 1M expert transitions in quite a short time. On the Humanoid and especially HalfCheetah especially we can see that the performance wasn't good. The reason might be that these environemnts are much more prone to failure (the robot is much more unstable) which means that a slight mistake in BC can end an episode.

Experiment configs and results:

HalfCheetah:

    Custom A2C:
    * config:
        --discount 0.96 -n 1000 -l 2 -s 128 -b 5000 -lr 0.003 --baseline_gradient_steps 10 -na -rtg --use_baseline --gae_lambda 0.97
    * runtime: 12 minutes
    * total env steps 5 000 000

    SB A2C:
    * config: same as above
    * runtime: 10 minutes

    Custom SAC:
    * config:
        --batch_size 256 --n_layers 3 --hidden_size 128 --learning_rate 3e-5 --total_steps 200000 --random_steps 5000 --discount 0.99 --soft_target_update_rate 0.004 --temperature 0.05 --activation relu
        
        --batch_size 256 --n_layers 2 --hidden_size 256 --learning_rate 4e-3 --alpha_learning_rate 2e-3 --total_steps 1000000 --random_steps 10000 --discount 0.98 --soft_target_update_rate 0.005 --temperature 0.05 --activation relu
    * runtime: 3 hours

    SB SAC:
    * config: same as above
    * runtime: 23 minutes (but only for 200k steps instead of 1M)

    Behavioral cloning:
    * config:
        --num_iter 10000 --train_batch_size 256 --n_layers 3 --hidden_size 256 --learning_rate 5e-4
    * runtime: 3 minutes
    * eval return: 470
    * expert data size: 1 000 000

Humanoid:

    Custom A2C:
    * config:
        --discount 0.99 --total_timesteps 5000000 -l 3 -s 256 -b 10000 -lr 0.001 --baseline_gradient_steps 10 -na -rtg --use_baseline --gae_lambda 0.97
    * runtime: 25 minutes
    * total env steps 5 000 000

    SB A2C:
    * config: same as above
    * runtime: 25 minutes

    Custom SAC:
    * config:
        --batch_size 256 --n_layers 3 --hidden_size 128 --learning_rate 3e-5 --total_steps 200000 --random_steps 5000 --discount 0.99 --soft_target_update_rate 0.004 --temperature 0.05 --activation relu
    * runtime: 6 hours

    SB SAC:
    * config: same as above
    * runtime: 3 hours

    Behavioral cloning:
    * config:
        --num_iter 10000 --train_batch_size 256 --n_layers 3 --hidden_size 256 --learning_rate 5e-4
    * runtime: 3 minutes
    * eval return: 470
    * expert data size: 999 434

Ant:

    Custom A2C:
    * config:
        --discount 0.98 --total_timesteps 5000000 -l 3 -s 256 -b 5000 -lr 0.002 --baseline_gradient_steps 10 -na -rtg --use_baseline --gae_lambda 0.97
    * runtime: 25 minutes
    * total env steps 5 000 000

    SB A2C:
    * config: same as above
    * runtime: 25 minutes

    Custom SAC:
    * config:
    --batch_size 256 --n_layers 2 --hidden_size 256 --learning_rate 4e-3 --total_steps 1000000 --random_steps 10000 --discount 0.98 --soft_target_update_rate 0.005 --temperature 0.05 --activation relu

    * runtime: 5 hours 30 minutes
    SB SAC:
    * config: same as above
    * runtime: 4 hours 30 minutes

    Behavioral cloning:
    * config:
        --num_iter 10000 --train_batch_size 256 --n_layers 3 --hidden_size 256 --learning_rate 5e-4
    * runtime: 3 minutes
    * eval return: 470
    * expert data size: 2 000 000