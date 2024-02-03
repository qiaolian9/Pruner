# Using Pruner on Nvidia A100  (or any other NV GPUs)
This is a quick tutorial on using Pruner on Nv A100(40G)

## Requirements
```
torch==1.8.1+cu111
torchvision==0.9.1
```
## Preparation
### Steps
1. Build and install this fork of TVM following the [guide](install/from_source.rst).

2. Register device abstraction on Pruner
```
# task_scheduler.py
psamodel_params = {
    "k80": dict(
        peak_performance=8226, glbmem_bandwidth=480, vec_len = 11,
        active_blocks_per_sm = 1, sm_nums = 26, arch_sm_partition = 4, arch_warp_size = 32
    ),
    "t4": dict(
        peak_performance=8141, glbmem_bandwidth=320, vec_len = 11,
        active_blocks_per_sm = 1, sm_nums = 40, arch_sm_partition = 4, arch_warp_size = 32
    ),
    "titanv": dict(
        peak_performance=14900, glbmem_bandwidth=651, vec_len = 11,
        active_blocks_per_sm = 1, sm_nums = 80, arch_sm_partition = 4, arch_warp_size = 32
    ),
    "orin": dict(
        peak_performance=2660, glbmem_bandwidth=204, vec_len = 11,
        active_blocks_per_sm = 1, sm_nums = 16, arch_sm_partition = 4, arch_warp_size = 32
    ),
    "a100": dict(
        peak_performance=19490, glbmem_bandwidth=1935, vec_len = 11,
        active_blocks_per_sm = 1, sm_nums = 108, arch_sm_partition = 4, arch_warp_size = 32
    ),
    "a100_40": dict(
        peak_performance=19490, glbmem_bandwidth=1555, vec_len = 11,
        active_blocks_per_sm = 1, sm_nums = 108, arch_sm_partition = 4, arch_warp_size = 32
    ),
}
```
##  on-line cost modelMode
### Steps
1. Search with Pruner w/o MTL (tuning 2,000 trials)
```bash
python3 tune_network.py --network resnet_50 --n-trials 2000 --cost-model pam --target "cuda --model=a100" --psa a100_40
```
Reference output: 
```
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.022 |         183.64 |     50 |
|    1 |        0.006 |          -0.00 |     10 |
|    2 |        0.022 |        4736.69 |     80 |
|    3 |        0.044 |        3228.89 |    120 |
|    4 |        0.044 |        2335.90 |    110 |
|    5 |        0.089 |        2600.16 |    110 |
|    6 |        0.040 |        5082.43 |     60 |
|    7 |        0.017 |        6122.10 |    130 |
|    8 |        0.035 |        3268.65 |    230 |
|    9 |        0.026 |        3979.41 |    160 |
|   10 |        0.058 |        4019.41 |     70 |
|   11 |        0.030 |        6922.56 |     50 |
|   12 |        0.014 |        7519.05 |     70 |
|   13 |        0.034 |        3686.11 |    140 |
|   14 |        0.019 |        5440.26 |     70 |
|   15 |        0.036 |        6437.31 |     50 |
|   16 |        0.023 |        9041.83 |     50 |
|   17 |        0.013 |        8188.52 |     50 |
|   18 |        0.028 |        4524.49 |    120 |
|   19 |        0.015 |        6868.58 |     50 |
|   20 |        0.008 |        3174.26 |     10 |
|   21 |        0.006 |         326.04 |     10 |
|   22 |        0.021 |       11279.94 |     30 |
|   23 |        0.013 |        8070.43 |     20 |
|   24 |        0.023 |        8825.72 |     30 |
|   25 |        0.035 |        5921.50 |     50 |
|   26 |        0.050 |        4080.21 |     70 |
-------------------------------------------------
Estimated total latency: 1.476 ms       Trials: 2000    Used time : 5563 s      Next ID: -1
same cost. continue
same cost. continue
same cost. continue
Mean inference time (std dev): 1.69 ms (0.00 ms)
```

2. Search with the Pruner (tuning 2,000 trials)
```bash
#  pam_k80_1500 trained from off-line cost's step 4
python3 tune_network.py --network resnet_50 --n-trials 2000 --cost-model pam-siamese-update --load-model pam_k80_1500.pkl --target "cuda --model=a100" --psa a100_40
```

Reference output:
```
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------                                  
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.022 |         184.55 |     30 |
|    1 |        0.006 |          -0.00 |     10 |
|    2 |        0.022 |        4730.83 |     90 |
|    3 |        0.047 |        3017.56 |    120 |
|    4 |        0.047 |        2172.09 |    120 |
|    5 |        0.089 |        2591.66 |    120 |
|    6 |        0.039 |        5268.03 |     60 |
|    7 |        0.016 |        6281.74 |    130 |
|    8 |        0.035 |        3307.53 |    230 |
|    9 |        0.027 |        3771.34 |    180 |
|   10 |        0.057 |        4072.68 |     80 |
|   11 |        0.028 |        7420.26 |     50 |
|   12 |        0.012 |        8429.32 |     70 |
|   13 |        0.032 |        3917.47 |    130 |
|   14 |        0.018 |        5647.25 |     80 |
|   15 |        0.029 |        7872.45 |     50 |
|   16 |        0.021 |        9699.48 |     30 |
|   17 |        0.012 |        8422.25 |     50 |
|   18 |        0.026 |        5026.50 |    100 |
|   19 |        0.016 |        6536.13 |     50 |
|   20 |        0.006 |        4617.30 |     10 |
|   21 |        0.006 |         323.04 |     10 |
|   22 |        0.021 |       11216.53 |     30 |
|   23 |        0.012 |        8366.00 |     20 |
|   24 |        0.023 |        8863.68 |     30 |
|   25 |        0.036 |        5689.20 |     50 |
|   26 |        0.050 |        4106.63 |     70 |
-------------------------------------------------
Estimated total latency: 1.457 ms       Trials: 2000    Used time : 4978 s      Next ID: -1
Mean inference time (std dev): 1.70 ms (0.00 ms)
```



3. Search with Ansor (tuning 2,000 trials)
```bash
# Using TensetMLP code
python3 tune_network.py --network resnet_50 --n-trials 2000 --cost-model mlp --target "cuda --model=a100"
```
Reference output: 
```
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.022 |         185.49 |     60 |
|    1 |        0.008 |          -0.00 |     10 |
|    2 |        0.028 |        3664.95 |    100 |
|    3 |        0.052 |        2743.14 |    110 |
|    4 |        0.042 |        2445.29 |    100 |
|    5 |        0.105 |        2199.96 |    140 |
|    6 |        0.045 |        4614.99 |     50 |
|    7 |        0.017 |        6118.53 |    120 |
|    8 |        0.044 |        2624.40 |    230 |
|    9 |        0.027 |        3773.70 |    160 |
|   10 |        0.052 |        4409.37 |     90 |
|   11 |        0.029 |        7205.22 |     30 |
|   12 |        0.015 |        7015.37 |     90 |
|   13 |        0.034 |        3767.54 |    110 |
|   14 |        0.021 |        4888.23 |     70 |
|   15 |        0.038 |        6140.99 |     50 |
|   16 |        0.022 |        9445.17 |     30 |
|   17 |        0.012 |        8548.53 |     50 |
|   18 |        0.031 |        4188.11 |    110 |
|   19 |        0.015 |        6847.98 |     50 |
|   20 |        0.008 |        3131.66 |     10 |
|   21 |        0.005 |         328.47 |     20 |
|   22 |        0.023 |       10309.79 |     30 |
|   23 |        0.012 |        8702.41 |     20 |
|   24 |        0.030 |        6934.80 |     50 |
|   25 |        0.033 |        6240.55 |     50 |
|   26 |        0.048 |        4250.69 |     60 |
-------------------------------------------------
Estimated total latency: 1.592 ms       Trials: 2000    Used time : 6691 s      Next ID: -1
('resnet_50', [(1, 3, 224, 224)])
Mean inference time (std dev): 3.96 ms (0.00 ms)
```

## off-line cost model Mode
### Steps

1. Download the GPU dataset (similar to Tenset).
- You can download it from google drive with the link [dataset_gpu_v3.3.zip](https://drive.google.com/file/d/1jqHbmvXUrLPDCIqJIaPee_atsPc0ZFFK/view?usp=sharing)
```bash
cd Pruner/scripts
pip3 install gdown
gdown https://drive.google.com/uc?id=1jqHbmvXUrLPDCIqJIaPee_atsPc0ZFFK
unzip dataset_gpu_v3.3.zip
ln -s dataset_gpu dataset
```

2. Sample a subset of the dataset and do featurization.
```bash
python make_dataset.py --logs ./dataset/measure_records/k80/*.json --hold-out all_five --out-file dataset_pam_k80_500.pkl --pam 1 --sample-in-files 500
```


3. Do featurization on your own target dataset. (similar to step.2)


4. Train a PAM model on TensetGPUs
```bash
python3 pruner_train_model.py --dataset dataset_pam_k80_500.pkl --model pam
```

Reference output:
```
Arguments: Namespace(dataset=['dataset_pam_k80_500.pkl'], file_cnt=0, fold=1, models='pam', ratio=1.0, save_path='./ckpt/k80/500', seed=0, select_cnt=0, split_scheme='within_task', train_ratio=0.9, use_gpu=False)
Load all tasks...
>> fold1
Load dataset...
Train set: 1121624. Task 0 = LearningTask(workload_key='["d3c2d35d319c1cb2e62f4f64aca23ad1", 4, 6, 6, 1024, 4, 4, 1024, 512, 1, 1, 1, 512, 1, 1, 1, 512, 4, 12, 12, 512]', target='cuda -keys=cuda,gpu -arch=sm_37 -max_num_threads=1024 -max_threads_per_block=1024 -registers_per_block=65536 -shared_memory_per_block=49152 -thread_warp_size=32')
Test set:  124626. Task 0 = LearningTask(workload_key='["d3c2d35d319c1cb2e62f4f64aca23ad1", 4, 6, 6, 1024, 4, 4, 1024, 512, 1, 1, 1, 512, 1, 1, 1, 512, 4, 12, 12, 512]', target='cuda -keys=cuda,gpu -arch=sm_37 -max_num_threads=1024 -max_threads_per_block=1024 -registers_per_block=65536 -shared_memory_per_block=49152 -thread_warp_size=32')
cuda:0
Save model to ./ckpt/k80/500/1/pam_k80_313_313.pkl
============================================================
Fit a net. Train size: 1121624
./ckpt/k80/500/1
Epoch: 0        Batch: 2190     Train Loss: 5.2418      Valid Loss: 77.2518     Train Speed: 28623      LR: 7.0000e-04
./ckpt/k80/500/1/epoch=0-loss=5.241805134825989-val_loss77.25178441693706.pkl
...
Epoch: 45       Batch: 2190     Train Loss: 1.3784      Valid Loss: 16.0521     Train Speed: 14029      LR: 7.0000e-04
./ckpt/k80/500/1/epoch=45-loss=1.3784006817091254-val_loss16.052119778048606.pkl
Test set sizes: [400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400,
400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 40
0, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400,
 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 4
00, 400, 400, 400, 400, 400, 400, 400, 400, 102, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400
, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 124, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400,
400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400]
pam {'RMSE': '11.925103', 'R^2': '-8128.657766', 'pairwise comparision accuracy': '0.903628', 'mape': '122978062983838.328125', 'average peak score@1': '0.906207', 'average peak score@5': '0.958415'}
------------------------------------------------------------
Model: pam
RMSE: 11.9251
R^2: -8128.6578
pairwise comparision accuracy: 0.9036
mape: 122978062983838.3281
average peak score@1: 0.9062
average peak score@5: 0.9584

```
5. Fine-tune a PAM model on target device (similar to step.4)
```bash
python3 fine_tune_model.py --dataset pam_a100_dataset.pkl --load-model pam_k80_1500.pkl
```


6. Search with the Pruner w/ finetuned model (tuning 2,000 trials)
```bash
python3 tune_network.py --network resnet_50 --n-trials 2000 --cost-model pam-no-update --load-model fine_tune_pam_a100.pkl --target "cuda --model=a100" --psa a100_40
```

Reference output:
```
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.022 |         185.52 |     50 |
|    1 |        0.006 |          -0.00 |     10 |
|    2 |        0.022 |        4742.03 |     80 |
|    3 |        0.049 |        2888.06 |    130 |
|    4 |        0.044 |        2334.09 |    110 |
|    5 |        0.090 |        2566.49 |    120 |
|    6 |        0.039 |        5233.33 |     50 |
|    7 |        0.017 |        6106.34 |    130 |
|    8 |        0.036 |        3223.69 |    220 |
|    9 |        0.029 |        3584.17 |    180 |
|   10 |        0.048 |        4862.99 |     60 |
|   11 |        0.025 |        8105.22 |     50 |
|   12 |        0.012 |        8415.31 |     70 |
|   13 |        0.031 |        4059.84 |    120 |
|   14 |        0.019 |        5495.82 |     80 |
|   15 |        0.035 |        6657.11 |     50 |
|   16 |        0.021 |        9893.88 |     30 |
|   17 |        0.012 |        8704.60 |     60 |
|   18 |        0.028 |        4667.28 |    110 |
|   19 |        0.015 |        6783.80 |     50 |
|   20 |        0.008 |        3292.55 |     10 |
|   21 |        0.006 |         328.39 |     10 |
|   22 |        0.022 |       10634.84 |     30 |
|   23 |        0.012 |        8513.47 |     20 |
|   24 |        0.026 |        7848.67 |     50 |
|   25 |        0.036 |        5713.19 |     50 |
|   26 |        0.049 |        4229.13 |     70 |
-------------------------------------------------
Estimated total latency: 1.469 ms       Trials: 2000    Used time : 4212 s      Next ID: -1
Mean inference time (std dev): 1.71 ms (0.00 ms)
```

6. Search with the TensetMLP (tuning 2,000 trials)
```bash
# using TensetMLP code
python3 tune_network.py --network resnet_50 --n-trials 2000 --cost-model mlp-no-update --load-model pam_finetune.pkl --target "cuda --model=a100"
```

Reference output:
```
----------------------------------------------------------------------
------------------------------  [ Task Scheduler ]
----------------------------------------------------------------------
|  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
-------------------------------------------------
|    0 |        0.023 |         182.03 |     30 |
|    1 |        0.006 |          -0.00 |     10 |
|    2 |        0.022 |        4745.88 |     80 |
|    3 |        0.045 |        3183.48 |    110 |
|    4 |        0.041 |        2509.52 |    100 |
|    5 |        0.090 |        2571.22 |    110 |
|    6 |        0.041 |        5059.50 |     60 |
|    7 |        0.018 |        5794.78 |    120 |
|    8 |        0.043 |        2678.46 |    250 |
|    9 |        0.035 |        2899.23 |    200 |
|   10 |        0.057 |        4035.98 |     70 |
|   11 |        0.032 |        6507.86 |     50 |
|   12 |        0.014 |        7508.15 |     70 |
|   13 |        0.035 |        3636.71 |    120 |
|   14 |        0.020 |        5089.64 |     70 |
|   15 |        0.043 |        5442.38 |     50 |
|   16 |        0.028 |        7501.13 |     50 |
|   17 |        0.013 |        8144.48 |     50 |
|   18 |        0.034 |        3761.48 |    120 |
|   19 |        0.019 |        5518.21 |     50 |
|   20 |        0.008 |        3387.65 |     10 |
|   21 |        0.006 |         323.08 |     10 |
|   22 |        0.023 |       10542.84 |     30 |
|   23 |        0.012 |        8907.67 |     20 |
|   24 |        0.023 |        8795.25 |     30 |
|   25 |        0.039 |        5256.55 |     60 |
|   26 |        0.058 |        3532.00 |     70 |
-------------------------------------------------
Estimated total latency: 1.621 ms       Trials: 2000    Used time : 5469 s      Next ID: -1
('resnet_50', [(1, 3, 224, 224)])
Mean inference time (std dev): 1.79 ms (0.00 ms)
```

## Summary
| method (tuning 2,000 trails)  | ansor | Pruner w/o MTL| Pruner | TensetMLP   | Pruner w/ finetuned model |
| ----------------- |  --- |  --- |--- | --- | --- |
| Update mode       | on-line| online| online| offline | offline |
| Search time(s)       | 6,691| 5,563 | 4,978 | 5,469 | 4,212|
| Estimated total latency(ms) | 1.592| 1.476| 1.457 |1.621 | 1.469|

- The Resnet-50's tuning curve with different method is shown as follows.
![R50_a100](./R50_A100_tuning_curve.png)


