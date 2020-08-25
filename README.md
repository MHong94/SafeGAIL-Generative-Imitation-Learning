# SafeGAIL

## basketball 

Code is written with PyTorch v0.4.1 (Python 3.6.5) based on [NAOMI](https://github.com/felixykliu/NAOMI). 
### To train the model:

First open visdom, then adjust hyperparameters in `train_model.sh` and run the shell file. And for forward prediction training do the same with `train_model_forward.sh`. Forward training needs trained model with random masking as initialization.

### Detailed explanations of hyperparameters:

•	`--model`: “NAOMI” or “SingleRes”

•	`--task`: “basketball” or “billiard”

•	`--y_dim`: 10 for basketball and 2 for billiard

•	`--rnn_dim` and `--n_layers`: gru cell size for all models, including forward and backward rnns

•	`--dec1_dim` to `--dec16_dim`: For NAOMI, these values correspond to dimensions of different decoders. For SingleRes, only dec1_dim is used for decoder.

•	`--pre_start_lr`: initial learning rate for supervised pretrain

•	`--pretrain`: supervised pretrain epochs

•	`--highest`: largest stepsize for NAOMI decoders, should be 2^n

•	`--discrim_rnn_dim` and `--discrim_layers`: discriminator rnn size

•	`--policy_learning_rate`: learning rate for generator in adversarial training

•	`--discrim_learning_rate`: learning rate for discriminator in adversarial training

•	`--pretrain_disc_iter`: number of iterations to pretrain discriminator

•	`--max_iter_num`: number of adversarial training iterations


## car racing

This is an implementation of Safe Generative Adversarial Imitation Learning (GAIL) for deterministic policies with off Policy learning on **static data**. **The policy never interacts with the environment** (except for evaluation), instead it is trained on policy state-action pair, where **policy only selects actions for states sampled from expert data**. 

For train - python train.py





