for seed in $(seq 1 5); do
PYTHONPATH=$PYTHONPATH:$(pwd) python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 \
--exp_name pendulum_exp4_s$seed \
-rtg --use_baseline -na \
--batch_size 1200 \
--discount 0.98 \
--n_layers 3 \
--layer_size 36 \
--gae_lambda 0.98 \
--seed $seed
done
