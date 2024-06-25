
# Pruner
A repo for Pruner

- This repo is based on a fork of [Tenset](https://github.com/tlc-pack/tenset).

## Installation

1. Build and install this repo following the [guide](https://github.com/AnonymousAsplos25/Pruner/blob/master/docs/install/from_source.rst).

2. Version information can refer to [here](requirements.txt).

3. Register device abstraction on Pruner as [abstraction](docs/try_Pruner_on_NvidiaA100.md)


## Using Pruner on Nvidia GPUs
Quick tutorial on using  Pruner on NVIDIA A100 refer to [tutorial](docs/try_Pruner_on_NvidiaA100.md).
###  On-Line cost model Mode
#### Steps
1. Search with Pruner (tuning 2,000 trials)
```
python3 tune_network.py --network resnet_50 --n-trials 2000 --cost-model pam --target "cuda --model=a100" --psa a100_40
```

2. Search with the MoA-Pruner (tuning 2,000 trials)
```
python3 tune_network.py --network resnet_50 --n-trials 2000 --cost-model pam-siamese-update --load-model pam_k80_1500.pkl --target "cuda --model=a100" --psa a100_40
```
3. Search with Ansor (tuning 2,000 trials)
```
# Using TensetMLP code
python3 tune_network.py --network resnet_50 --n-trials 2000 --cost-model mlp --target "cuda --model=a100"
```

### Off-line cost model Mode
#### Steps
1. Pretrain or Finetune a model refer to [tutorial](docs/try_Pruner_on_NvidiaA100.md)



2. Search with the Pruner w/ finetuned model (tuning 2,000 trials)
```
python3 tune_network.py --network resnet_50 --n-trials 2000 --cost-model pam-no-update --load-model pam_finetune.pkl --target "cuda --model=a100" --psa a100_40
```

3. Search with the TensetMLP (tuning 2,000 trials)
```
# Using TensetMLP code
python3 tune_network.py --network resnet_50 --n-trials 2000 --cost-model mlp-no-update --load-model mlp_finetune.pkl --target "cuda --model=a100"
```

### Summary
| method (tuning 2,000 trails)  | ansor | Pruner | MoA-Pruner | TensetMLP   | Pruner w/ finetuned model |
| ----------------- |  --- |  --- |--- | --- | --- |
| Update mode       | on-line| online| online| offline | offline |
| Search time(s)       | 6,691| 5,563 | 4,978 | 5,469 | 4,212|
| Estimated total latency(ms) | 1.592| 1.476| 1.457 |1.621 | 1.469|

Note: Details are reported in './scripts/res/resnet_50'

- The Resnet-50's tuning curve with different method is shown as follows.

![R50_a100](./docs/R50_A100_tuning_curve.png)


### Tuning result for end-to-end workload benchmark
-  Detiled tuning results refer to [E2E_Tuning_Comparison](docs/Pruner_e2e_tuning_Comparison.md)

- The following figure shows that the search time required for Pruner to reach the performance of different approach tuning 2,000 trials on A100.

![compilertime_a100](./docs/Prunerresults/compilertime_a100.png)


