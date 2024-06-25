"""Make a dataset file.

Usage:
python make_dataset.py --logs ./dataset/measure_records/k80/*.json --hold-out all_five --out-file dataset_pam_k80_all.pkl --pam 1 --sample-in-files 1750
"""
import argparse
import glob
import pickle
import random

from tqdm import tqdm
import tvm
from tvm import auto_scheduler

from common import (load_and_register_tasks, get_task_info_filename,
    get_measure_record_filename)

from dump_network_info import build_network_keys

import numpy as np
import os

from tvm.auto_scheduler import RecordReader, MeasureInput, MeasureResult
from tvm.auto_scheduler.feature import get_per_store_features_from_measure_pairs_psa

from dump_network_info import build_network_keys

from collections import namedtuple, OrderedDict, defaultdict
LearningTask = namedtuple("LearningTask", ['workload_key', 'target'])


def input_to_learning_task(inp: MeasureInput):
    return LearningTask(inp.task.workload_key, str(inp.task.target))

def make_psa_dataset_from_log_file(log_files, out_file, min_sample_size, verbose=1):
    """Make a dataset file from raw log files"""
    from tqdm import tqdm

    cache_folder = ".dataset_cache"
    os.makedirs(cache_folder, exist_ok=True)

    dataset = auto_scheduler.dataset.Dataset()
    dataset.raw_files = log_files
    for filename in tqdm(log_files):
        assert os.path.exists(filename), f"{filename} does not exist."

        cache_file = f"{cache_folder}/{filename.replace('/', '_')}_psa.feature_cache"
        if os.path.exists(cache_file):
            # Load feature from the cached file
            features, throughputs, min_latency = pickle.load(open(cache_file, "rb"))
        else:
            # Read measure records
            measure_records = {}
            for inp, res in RecordReader(filename):
                task = input_to_learning_task(inp)
                if task not in measure_records:
                    measure_records[task] = [[], []]
                measure_records[task][0].append(inp)
                measure_records[task][1].append(res)
            # Featurize
            features = {}
            throughputs = {}
            min_latency = {}
            for task, (inputs, results) in measure_records.items():
                features_, normalized_throughputs, task_ids, min_latency_ =\
                    get_per_store_features_from_measure_pairs_psa(inputs, results)

                assert not np.any(task_ids)   # all task ids should be zero
                if len(min_latency_) == 0:
                    # no valid records
                    continue
                else:
                    # should have only one task
                    assert len(min_latency_) == 1, f"len = {len(min_latency)} in {filename}"

                features[task] = features_
                throughputs[task] = normalized_throughputs
                min_latency[task] = min_latency_[0]
            pickle.dump((features, throughputs, min_latency), open(cache_file, "wb"))

        for task in features:
            dataset.load_task_data(task, features[task], throughputs[task], min_latency[task])

    # Delete task with too few samples
    to_delete = []
    for i, (task, feature) in enumerate(dataset.features.items()):
        if verbose >= 0:
            print("No: %d\tTask: %s\tSize: %d" % (i, task, len(feature)))
        if len(feature) < min_sample_size:
            if verbose >= 0:
                print("Deleted")
            to_delete.append(task)
    for task in to_delete:
        del dataset.features[task]
        del dataset.throughputs[task]
        del dataset.min_latency[task]

    # Save to disk
    pickle.dump(dataset, open(out_file, "wb"))

    if verbose >= 0:
        print("A dataset file is saved to %s" % out_file)

def get_hold_out_task(target, network=None):
    network_keys = []

    if network == "resnet-50":
        print("precluding all tasks in resnet-50")
        for batch_size in [1, 4, 8]:
            for image_size in [224, 240, 256]:
                for layer in [50]:
                    network_keys.append((f'resnet_{layer}',
                                         [(batch_size, 3, image_size, image_size)]))
    else:
        # resnet_18 and resnet_50
        for layer in [18, 50]:
            network_keys.append((f'resnet_{layer}', [(1, 3, 224, 224)]))

        # mobilenet_v2
        network_keys.append(('mobilenet_v2', [(1, 3, 224, 224)]))

        # resnext
        network_keys.append(('resnext_50', [(1, 3, 224, 224)]))

        # bert
        for scale in ['tiny', 'base']:
            network_keys.append((f'bert_{scale}', [(1, 128)]))

    exists = set()
    print("hold out...")
    for network_key in tqdm(network_keys):
        # Read tasks of the network
        task_info_filename = get_task_info_filename(network_key, target)
        tasks, _ = pickle.load(open(task_info_filename, "rb"))
        for task in tasks:
            if task.workload_key not in exists:
                exists.add(task.workload_key)

    return exists


def preset_batch_size_1():
    network_keys = []

    # resnet_18 and resnet_50
    for batch_size in [1]:
        for image_size in [224, 240, 256]:
            for layer in [18, 50]:
                network_keys.append((f'resnet_{layer}',
                                    [(batch_size, 3, image_size, image_size)]))

    # mobilenet_v2
    for batch_size in [1]:
        for image_size in [224, 240, 256]:
            for name in ['mobilenet_v2', 'mobilenet_v3']:
                network_keys.append((f'{name}',
                                    [(batch_size, 3, image_size, image_size)]))

    # wide-resnet
    for batch_size in [1]:
        for image_size in [224, 240, 256]:
            for layer in [50]:
                network_keys.append((f'wide_resnet_{layer}',
                                    [(batch_size, 3, image_size, image_size)]))

    # resnext
    for batch_size in [1]:
        for image_size in [224, 240, 256]:
            for layer in [50]:
                network_keys.append((f'resnext_{layer}',
                                    [(batch_size, 3, image_size, image_size)]))

    # inception-v3
    for batch_size in [1]:
        for image_size in [299]:
            network_keys.append((f'inception_v3',
                                [(batch_size, 3, image_size, image_size)]))

    # densenet
    for batch_size in [1]:
        for image_size in [224, 240, 256]:
            network_keys.append((f'densenet_121',
                                [(batch_size, 3, image_size, image_size)]))

    # resnet3d
    for batch_size in [1]:
        for image_size in [112, 128, 144]:
            for layer in [18]:
                network_keys.append((f'resnet3d_{layer}',
                                    [(batch_size, 3, image_size, image_size, 16)]))

    # bert
    for batch_size in [1]:
        for seq_length in [64, 128, 256]:
            for scale in ['tiny', 'base', 'medium', 'large']:
                network_keys.append((f'bert_{scale}',
                                    [(batch_size, seq_length)]))

    # dcgan
    for batch_size in [1]:
        for image_size in [64, 80, 96]:
            network_keys.append((f'dcgan',
                                [(batch_size, 3, image_size, image_size)]))

    return network_keys

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

set_seed(0)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", type=str)
    parser.add_argument("--target", nargs="+", type=str, default=["cuda --model=k80 -keys=cuda,gpu -arch=sm_75 -max_num_threads=1024 -max_threads_per_block=1024 -registers_per_block=65536 -shared_memory_per_block=49152 -thread_warp_size=32"])
    parser.add_argument("--sample-in-files", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-file", type=str, default='dataset.pkl')
    parser.add_argument("--min-sample-size", type=int, default=1000)
    parser.add_argument("--hold-out", type=str, choices=['resnet-50', 'all_five'])
    parser.add_argument("--preset", type=str, choices=['batch-size-1'])
    parser.add_argument("--n-task", type=int)
    parser.add_argument("--n-measurement", type=int)
    parser.add_argument("--pam", type=int)

    args = parser.parse_args()

    random.seed(args.seed)

    files = []
    if args.hold_out or args.n_task:
        task_cnt = 0
        for target in args.target:
            target = tvm.target.Target(target)
            to_be_excluded = get_hold_out_task(target, args.hold_out)
            network_keys = build_network_keys()

            print("Load tasks...")
            print(f"target: {target}")
            all_tasks = []
            exists = set()  # a set to remove redundant tasks
            for network_key in tqdm(network_keys):
                # Read tasks of the network
                task_info_filename = get_task_info_filename(network_key, target)
                tasks, _ = pickle.load(open(task_info_filename, "rb"))
                for task in tasks:
                    if task.workload_key not in to_be_excluded and task.workload_key not in exists:
                        if not args.n_task or task_cnt < args.n_task:
                            exists.add(task.workload_key)
                            all_tasks.append(task)
                            task_cnt += 1

            # Convert tasks to filenames
            for task in all_tasks:
                filename = get_measure_record_filename(task, target)
                files.append(filename)
    elif args.preset == 'batch-size-1':
        # Only use tasks from networks with batch-size = 1

        # Load tasks from networks
        network_keys = preset_batch_size_1()
        target = tvm.target.Target(args.target[0])
        all_tasks = []
        exists = set()   # a set to remove redundant tasks
        print("Load tasks...")
        for network_key in tqdm(network_keys):
            # Read tasks of the network
            task_info_filename = get_task_info_filename(network_key, target)
            tasks, _ = pickle.load(open(task_info_filename, "rb"))
            for task in tasks:
                if task.workload_key not in exists:
                    exists.add(task.workload_key)
                    all_tasks.append(task)

        # Convert tasks to filenames
        files = []
        for task in all_tasks:
            filename = get_measure_record_filename(task, target)
            files.append(filename)
    else:
        # use all tasks
        print("Load tasks...")
        load_and_register_tasks()
        files = args.logs

    if args.sample_in_files:
        files = random.sample(files, args.sample_in_files)

    print(">> file cnt\t#", len(files))
    print("Featurize measurement records...")
    if args.pam == 1:
        args.pam = True
    else:
        args.pam = False
    print(args.pam)
    if args.pam == True:
        print("pam true")
    else:
        print("pam false")
        
    auto_scheduler.dataset.make_dataset_from_log_file(
        files, args.out_file, args.min_sample_size, pam=args.pam)

