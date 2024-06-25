"""Evaluate a cost model on a network with dataset simulator"""
import argparse
import os
import pickle

import numpy as np

import tvm
from tvm import auto_scheduler
from tvm.auto_scheduler.utils import to_str_round
from tvm.auto_scheduler.dataset import LearningTask
from tvm.auto_scheduler.cost_model.psa_model import PSAModelInternal
from tvm.auto_scheduler.cost_model import RandomModelInternal

from common import get_task_info_filename, get_measure_record_filename
from train_model import evaluate_model
from make_dataset import make_psa_dataset_from_log_file


space_size = [256]
best_ks = [1, 5, 20]
def eval_cost_model_on_weighted_tasks(model, eval_task_dict, eval_dataset, space_size):
    """Evaluate a cost model on weighted tasks"""
    best_latency = 0
    # repeat_num = 1
    if isinstance(model, PSAModelInternal):
        repeat_num = 1
    elif isinstance(model, RandomModelInternal):
        repeat_num = 5000
    latencies = [0] * len(best_ks)
    for idx in range(repeat_num):
        preds_dict = model.predict(eval_dataset)

        for task, weight in eval_task_dict.items():
            if task not in eval_dataset.throughputs:
                print(task)
                print(f"Warning: cannot find {task.workload_key} in the eval_dataset. Skipped.")
                continue

            preds = preds_dict[task]
            labels, min_latency = eval_dataset.throughputs[task], eval_dataset.min_latency[task]

            real_values = labels[np.argsort(-preds)]
            real_latency = min_latency / np.maximum(real_values, 1e-5)

            # analysis 1
            population_sample = real_latency[:space_size[0]]
            population_sample = np.sort(population_sample)
            # print("*"* 10, f"{task}-{len(preds)}", "*" * 10)
            for i, best_k in enumerate(best_ks):
                latencies[i] += population_sample[best_k-1] * weight
            

            if idx == 0:
                best_latency += min_latency * weight

    for i in range(len(best_ks)):
        latencies[i] /= repeat_num

    return latencies, best_latency


def eval_cost_model_on_network(model, network_key, target, space_size):
    # Read tasks of the network
    target = tvm.target.Target(target)
    task_info_filename = get_task_info_filename(network_key, target)
    tasks, task_weights = pickle.load(open(task_info_filename, "rb"))
    network_task_key2 = (network_key, str(target))

    # Featurizes a dataset 
    dataset_file = f".dataset_cache/{network_task_key2}.network.psa_feature_cache"
    if not os.path.exists(dataset_file):
        # get file names of these tasks
        filenames = []
        for task in tasks:
            filename = get_measure_record_filename(task, target)
            filenames.append(filename)

        # make a dataset
        make_psa_dataset_from_log_file(
            filenames, dataset_file, min_sample_size=0)
    dataset = pickle.load(open(dataset_file, "rb"))
    
    eval_res = evaluate_model(model, dataset)
    print(to_str_round(eval_res))
    print("===============================================")

    target = targer_info[target.model]
    # learning_tasks = [LearningTask(t.workload_key, target) for t in tasks]
    learning_tasks = []
    for t in tasks:
        if LearningTask(t.workload_key, str(t.target)) in dataset.throughputs:
            learning_tasks.append(LearningTask(t.workload_key, str(t.target)))
        else:
            learning_tasks.append(LearningTask(t.workload_key, target))
    task_dict = {task: weight for task, weight in zip(learning_tasks, task_weights)}

    return eval_cost_model_on_weighted_tasks(model, task_dict, dataset, space_size)

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
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-model", type=str)
    args= parser.parse_args()

    network_keys = [
        ("resnet_50", [(1, 3, 224,224)]),
        ("mobilenet_v2", [(1, 3, 224,224)]),
        ("resnet3d_18", [(1,3,112,112,16)]),
        ("bert_base", [(1, 128)]),
        ("bert_tiny", [(1, 128)]),
    ]

    targer_info = {
        't4': 'cuda -keys=cuda,gpu -arch=sm_37 -max_num_threads=1024 -max_threads_per_block=1024 -registers_per_block=65536 -shared_memory_per_block=49152 -thread_warp_size=32',
        'k80': 'cuda -keys=cuda,gpu -arch=sm_37 -max_num_threads=1024 -max_threads_per_block=1024 -registers_per_block=65536 -shared_memory_per_block=49152 -thread_warp_size=32'
    }
    target = f"cuda -model={args.cuda_model}"

    model = PSAModelInternal(**psamodel_params[args.cuda_model])
    # model = RandomModelInternal()

    best_1_total = []
    best_5_total = []
    best_20_total = []
    for network_key in network_keys:
        latencies, best_latency = eval_cost_model_on_network(model, network_key, target, space_size)
        for top_k, latency in zip(space_size, latencies):
            print(f"Network: {network_key}\tTop-{top_k} score: {best_latency / latency}")

        best_1_total.append(best_latency/latencies[0])
        print(f"top {best_ks[0]} score: {best_latency/latencies[0]}")
        best_5_total.append(best_latency / latencies[1])
        print(f"top {best_ks[1]} score: {best_latency / latencies[1]}")
        best_20_total.append(best_latency/latencies[2])
        print(f"top {best_ks[2]} score: {best_latency/latencies[2]}")

    print(f"average best {best_ks[0]} score is {sum(best_1_total) / len(best_1_total)}")
    print(f"average best {best_ks[1]} score is {sum(best_5_total) / len(best_5_total)}")
    print(f"average best {best_ks[2]} score is {sum(best_20_total) / len(best_20_total)}")




