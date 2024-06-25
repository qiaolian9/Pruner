"""Dataset management"""
from collections import namedtuple, OrderedDict, defaultdict
import os
import pickle
from typing import List, Tuple

import numpy as np

from .measure_record import RecordReader
from .measure import MeasureInput, MeasureResult
from .feature import get_per_store_features_from_measure_pairs, get_per_store_features_from_measure_pairs_pam
import math
LearningTask = namedtuple("LearningTask", ['workload_key', 'target'])


def input_to_learning_task(inp: MeasureInput):
    return LearningTask(inp.task.workload_key, str(inp.task.target))


DATASET_FORMAT_VERSION = 0.1


class Dataset:
    def __init__(self):
        self.raw_files = None

        self.features = OrderedDict()      # Dict[LearningTask -> feature]
        self.throughputs = OrderedDict()   # Dict[LearningTask -> normalized_throughputs]
        self.min_latency = {}              # Dict[LearningTask -> min latency]
        self.measure_records = {}          # Dict[LearningTask -> Tuple[List[MeasureInput], List[MeasureResult]]

    @staticmethod
    def create_one_task(task, features, throughputs, min_latency=None):
        """Create a new dataset with one task and its feature and throughput data"""
        ret = Dataset()
        ret.load_task_data(task, features, throughputs, min_latency)
        return ret

    def update_from_measure_pairs(self, inputs: List[MeasureInput], results: List[MeasureResult]):
        new_data = {}  # Dict[LearningTask -> Tuple[List[MeasureInput], List[MeasureResult]]]
        for inp, res in zip(inputs, results):
            learning_task = input_to_learning_task(inp)
            store_tuple = new_data.get(learning_task, None)
            if store_tuple is None:
                store_tuple = ([], [])
                new_data[learning_task] = store_tuple
            store_tuple[0].append(inp)
            store_tuple[1].append(res)

        for task, (inputs, results) in new_data.items():
            features, normalized_throughputs, task_ids, min_latency =\
                get_per_store_features_from_measure_pairs(inputs, results)

            assert not np.any(task_ids)   # all task ids should be zero
            assert len(min_latency) == 1  # should have only one task

            self.load_task_data(task, features, normalized_throughputs, min_latency[0])

    def update_from_dataset(self, dataset):
        for task in dataset.features:
            if task not in self.features:
                self.features[task] = dataset.features[task]
                self.throughputs[task] = dataset.throughputs[task]
                self.min_latency[task] = dataset.min_latency[task]

    def load_task_data(self, task: LearningTask, features, throughputs, min_latency=None):
        """Load feature and throughputs for one task"""
        if task not in self.features:
            self.features[task] = features
            self.throughputs[task] = throughputs
            self.min_latency[task] = min_latency
        else:
            try:
                self.features[task] = np.concatenate([self.features[task], features])
            except ValueError:
                # Fix the problem of shape mismatch
                new_features = list(self.features[task])
                new_features.extend(features)
                self.features[task] = np.array(new_features, dtype=object)
            assert min_latency is not None
            combined_min_latency = min(self.min_latency[task], min_latency)
            self.throughputs[task] = np.concatenate([
                self.throughputs[task] * (combined_min_latency / self.min_latency[task]),
                throughputs * (combined_min_latency / min_latency)])
            self.min_latency[task] = combined_min_latency

    def random_split_within_task(self,
                                 train_set_ratio: float=None,
                                 train_set_num: int=None,
                                 shuffle_time: bool=False) -> Tuple["Dataset", "Dataset"]:
        """Randomly split the dataset into a training set and a test set.
        Do the split within each task. A measurement record is a basic unit.
        """
        train_set = Dataset()
        test_set = Dataset()

        assert train_set_ratio is not None or train_set_num is not None

        for task in self.features:
            features, throughputs = self.features[task], self.throughputs[task]
            if train_set_num is None:
                split = int(train_set_ratio * len(features))
            else:
                split = train_set_num

            if shuffle_time:
                perm = np.random.permutation(len(features))
                train_indices, test_indices = perm[:split], perm[split:]
            else:
                arange = np.arange(len(features))
                arange = np.flip(arange)
                train_indices, test_indices = arange[:split], arange[split:]

            if len(train_indices):
                train_throughputs = throughputs[train_indices]
                train_min_latency = self.min_latency[task] / np.max(train_throughputs)
                train_set.load_task_data(task, features[train_indices], train_throughputs, train_min_latency)

            if len(test_indices):
                test_throughputs = throughputs[test_indices]
                test_min_latency = self.min_latency[task] / np.max(test_throughputs)
                test_set.load_task_data(task, features[test_indices], test_throughputs, test_min_latency)

        return train_set, test_set

    def random_split_by_task(self, train_set_ratio: float) -> Tuple["Dataset", "Dataset"]:
        """Randomly split the dataset into a training set and a test set.
        Split tasks into two sets. A learning task is a basic unit.
        """
        tasks = list(self.features.keys())
        np.random.shuffle(tasks)

        train_records = int(len(self) * train_set_ratio)

        train_set = Dataset()
        test_set = Dataset()
        ct = 0
        for task in tasks:
            features, throughputs = self.features[task], self.throughputs[task]
            ct += len(features)
            if ct <= train_records:
                train_set.load_task_data(task, features, throughputs, self.min_latency[task])
            else:
                test_set.load_task_data(task, features, throughputs, self.min_latency[task])

        return train_set, test_set

    def random_split_by_target(self, train_set_ratio: float) -> Tuple["Dataset", "Dataset"]:
        """Randomly split the dataset into a training set and a test set.
        Split targets into two sets. A target is a basic unit.
        """
        target_to_task = defaultdict(list)
        for task in self.features.keys():
            target_to_task[str(task.target)].append(task)
        targets = list(target_to_task.keys())
        targets = list(reversed(targets))
        #np.random.shuffle(targets)

        train_records = int(len(self) * train_set_ratio)

        train_set = Dataset()
        test_set = Dataset()
        ct = 0
        for target in targets:
            tmp_adder = 0
            for task in target_to_task[target]:
                features, normalized_throughputs = self.features[task], self.throughputs[task]
                tmp_adder += len(features)
                if ct <= train_records:
                    train_set.load_task_data(task, features, normalized_throughputs)
                else:
                    test_set.load_task_data(task, features, normalized_throughputs)
            ct += tmp_adder

        return train_set, test_set

    def tasks(self) -> List[LearningTask]:
        """Get all tasks"""
        if self.features:
            return list(self.features.keys())
        else:
            return list(self.measure_records.keys())

    def targets(self) -> List[str]:
        """Get all targest"""
        ret = set()
        for t in self.tasks():
            ret.add(t.target)
        return list(ret)

    def extract_subset(self, tasks: List[LearningTask]) -> "Dataset":
        """Extract a subset containing given tasks"""
        ret = Dataset()
        for task in tasks:
            if not (task in self.features):
                continue
            ret.load_task_data(task, self.features[task], self.throughputs[task], self.min_latency[task])
        return ret

    def __getstate__(self):
        return self.raw_files, self.features, self.throughputs, self.min_latency, DATASET_FORMAT_VERSION

    def __setstate__(self, value):
        self.raw_files, self.features, self.throughputs, self.min_latency, format_version = value

    def __len__(self, ):
        return sum(len(x) for x in self.throughputs.values())


class PAMDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.features_sizes = OrderedDict()   
        self.kmp_indexs = OrderedDict()   
        self.buf_features = OrderedDict()   
       

    @staticmethod
    def create_one_task(task, features, buf_features, features_sizes, kmp_indexs, throughputs, min_latency=None):
        """Create a new dataset with one task and its feature and throughput data"""
        ret = PAMDataset()
        ret.load_task_data(task, features, buf_features, features_sizes, kmp_indexs, throughputs, min_latency)
        return ret
    
    def update_from_measure_pairs(self, inputs: List[MeasureInput], results: List[MeasureResult]):
        new_data = {}  # Dict[LearningTask -> Tuple[List[MeasureInput], List[MeasureResult]]]
        for inp, res in zip(inputs, results):
            learning_task = input_to_learning_task(inp)
            store_tuple = new_data.get(learning_task, None)
            if store_tuple is None:
                store_tuple = ([], [])
                new_data[learning_task] = store_tuple
            store_tuple[0].append(inp)
            store_tuple[1].append(res)

        for task, (inputs, results) in new_data.items():
            features, buf_features, features_sizes, kmp_indexs, normalized_throughputs, task_ids, min_latency =\
                get_per_store_features_from_measure_pairs_pam(inputs, results, mode=False)

            assert not np.any(task_ids)   # all task ids should be zero
            assert len(min_latency) == 1  # should have only one task

            self.load_task_data(task, features, buf_features, features_sizes, kmp_indexs, normalized_throughputs, min_latency[0])

    def update_from_dataset(self, dataset):
        for task in dataset.features:
            if task not in self.features:
                self.features[task] = dataset.features[task]
                self.buf_features[task] = dataset.buf_features[task]
                self.features_sizes[task] = dataset.features_sizes[task]
                self.kmp_indexs[task] = dataset.kmp_indexs[task]
                self.throughputs[task] = dataset.throughputs[task]
                self.min_latency[task] = dataset.min_latency[task]

    def load_task_data(self, task: LearningTask, features, buf_features, features_sizes, kmp_indexs, throughputs, min_latency=None):
        """Load feature and throughputs for one task"""
        if task not in self.features:
            self.features[task] = features
            self.buf_features[task] = buf_features
            self.features_sizes[task] = features_sizes
            self.kmp_indexs[task] = kmp_indexs
            self.throughputs[task] = throughputs
            self.min_latency[task] = min_latency
        else:
            try:
                self.features[task] = np.concatenate([self.features[task], features])
                self.buf_features[task] = np.concatenate([self.buf_features[task], buf_features])
                self.features_sizes[task] = np.concatenate([self.features_sizes[task], features_sizes])
                self.kmp_indexs[task] = np.concatenate([self.kmp_indexs[task], kmp_indexs])
            except ValueError:
                # Fix the problem of shape mismatch
                new_features = list(self.features[task])
                new_features.extend(features)
                self.features[task] = np.array(new_features, dtype=object)
                new_buf_features = list(self.buf_features[task])
                new_buf_features.extend(buf_features)
                self.buf_features[task] = np.array(new_buf_features, dtype=object)
                new_features_sizes = list(self.features_sizes[task])
                new_features_sizes.extend(features_sizes)
                self.features_sizes[task] = np.array(new_features_sizes, dtype=object)
                new_kmp_indexs = list(self.kmp_indexs[task])
                new_kmp_indexs.extend(kmp_indexs)
                self.kmp_indexs[task] = np.array(new_kmp_indexs, dtype=object)
            assert min_latency is not None
            combined_min_latency = min(self.min_latency[task], min_latency)
            self.throughputs[task] = np.concatenate([
                self.throughputs[task] * (combined_min_latency / self.min_latency[task]),
                throughputs * (combined_min_latency / min_latency)])
            self.min_latency[task] = combined_min_latency

    def random_split_within_task(self,
                                 train_set_ratio: float=None,
                                 train_set_num: int=None,
                                 shuffle_time: bool=False) -> Tuple["PAMDataset", "PAMDataset"]:
        """Randomly split the dataset into a training set and a test set.
        Do the split within each task. A measurement record is a basic unit.
        """
        train_set = PAMDataset()
        test_set = PAMDataset()

        assert train_set_ratio is not None or train_set_num is not None

        for task in self.features:
            features, buf_features, features_sizes, kmp_indexs, throughputs = self.features[task], self.buf_features[task], self.features_sizes[task], self.kmp_indexs[task], self.throughputs[task]
            if train_set_num is None:
                split = int(train_set_ratio * len(features))
            else:
                split = train_set_num

            if shuffle_time:
                perm = np.random.permutation(len(features))
                train_indices, test_indices = perm[:split], perm[split:]
            else:
                arange = np.arange(len(features))
                arange = np.flip(arange)
                train_indices, test_indices = arange[:split], arange[split:]

            if len(train_indices):
                train_throughputs = throughputs[train_indices]
                train_min_latency = self.min_latency[task] / np.max(train_throughputs)
                train_set.load_task_data(task, features[train_indices], buf_features[train_indices], features_sizes[train_indices], kmp_indexs[train_indices], train_throughputs, train_min_latency)

            if len(test_indices):
                test_throughputs = throughputs[test_indices]
                test_min_latency = self.min_latency[task] / np.max(test_throughputs)
                test_set.load_task_data(task, features[test_indices], buf_features[test_indices], features_sizes[test_indices], kmp_indexs[test_indices], test_throughputs, test_min_latency)

        return train_set, test_set

    def random_split_by_task(self, train_set_ratio: float) -> Tuple["PAMDataset", "PAMDataset"]:
        """Randomly split the dataset into a training set and a test set.
        Split tasks into two sets. A learning task is a basic unit.
        """
        tasks = list(self.features.keys())
        np.random.shuffle(tasks)

        train_records = int(len(self) * train_set_ratio)

        train_set = PAMDataset()
        test_set = PAMDataset()
        ct = 0
        for task in tasks:
            features, buf_features, features_sizes, kmp_indexs, throughputs = self.features[task], self.buf_features[task], self.features_sizes[task], self.kmp_indexs[task], self.throughputs[task]
            ct += len(features)
            if ct <= train_records:
                train_set.load_task_data(task, features, buf_features, features_sizes, kmp_indexs, throughputs, self.min_latency[task])
            else:
                test_set.load_task_data(task, features, buf_features, features_sizes, kmp_indexs, throughputs, self.min_latency[task])

        return train_set, test_set

    def random_split_by_target(self, train_set_ratio: float) -> Tuple["PAMDataset", "PAMDataset"]:
        """Randomly split the dataset into a training set and a test set.
        Split targets into two sets. A target is a basic unit.
        """
        target_to_task = defaultdict(list)
        for task in self.features.keys():
            target_to_task[str(task.target)].append(task)
        targets = list(target_to_task.keys())
        targets = list(reversed(targets))
        #np.random.shuffle(targets)

        train_records = int(len(self) * train_set_ratio)

        train_set = PAMDataset()
        test_set = PAMDataset()
        ct = 0
        for target in targets:
            tmp_adder = 0
            for task in target_to_task[target]:
                features, buf_features, features_sizes, kmp_indexs, normalized_throughputs = self.features[task], self.buf_features[task], self.features_sizes[task], self.kmp_indexs[task], self.throughputs[task]
                tmp_adder += len(features)
                if ct <= train_records:
                    train_set.load_task_data(task, features, buf_features, features_sizes, kmp_indexs, normalized_throughputs)
                else:
                    test_set.load_task_data(task, features, buf_features, features_sizes, kmp_indexs, normalized_throughputs)
            ct += tmp_adder

        return train_set, test_set
    
    def extract_subset_ratio(self, ratio=0.1) -> "PAMDataset":
        """Extract a subset containing given tasks"""
        ret = PAMDataset()
        for task in self.features.keys():
            record_num = len(self.features[task])
            select_num = math.floor(ratio * record_num)
            if select_num <= 10:
                select_num = record_num
            # print(select_num)
            perm = np.random.permutation(record_num)
            random_select = perm[:select_num]
            record_features = self.features[task][random_select]
            record_buf_features = self.buf_features[task][random_select]
            record_features_sizes = self.features_sizes[task][random_select]
            record_kmp_indexs = self.kmp_indexs[task][random_select]
            record_throughputs = self.throughputs[task][random_select]
            record_min_latency = self.min_latency[task]
            ret.load_task_data(task, record_features, record_buf_features, record_features_sizes, record_kmp_indexs, record_throughputs, record_min_latency)
        return ret


    def extract_subset(self, tasks: List[LearningTask]) -> "PAMDataset":
        """Extract a subset containing given tasks"""
        ret = PAMDataset()
        for task in tasks:
            if not (task in self.features):
                continue
            ret.load_task_data(task, self.features[task], self.buf_features[task], self.features_sizes[task], self.kmp_indexs[task], self.throughputs[task], self.min_latency[task])
        return ret

    def __getstate__(self):
        return self.raw_files, self.features, self.buf_features, self.features_sizes, self.kmp_indexs, self.throughputs, self.min_latency, DATASET_FORMAT_VERSION

    def __setstate__(self, value):
        self.raw_files, self.features, self.buf_features, self.features_sizes, self.kmp_indexs, self.throughputs, self.min_latency, format_version = value

    def gemmtasks(self) -> List[LearningTask]:
        """Get all gemm tasks"""
        gemm_list = []
        for i in self.features.keys():
            fea_size = self.features_sizes[i]
            kmp_index = self.kmp_indexs[i]
            if max(fea_size) > 10 and max(kmp_index) > -1:
                gemm_list.append(i)

        return gemm_list

        


def make_dataset_from_log_file(log_files, out_file, min_sample_size, verbose=1, pam=False):
    """Make a dataset file from raw log files"""
    from tqdm import tqdm

    cache_folder = ".dataset_cache"
    os.makedirs(cache_folder, exist_ok=True)

    if pam == True:
        print('pam Dataset')
        mode = 'pam'
        dataset = PAMDataset()
    else:
        print('ansor Dataset')
        mode = 'ansor'
        dataset = Dataset()
    dataset.raw_files = log_files
    for filename in tqdm(log_files):
        try:
            assert os.path.exists(filename), f"{filename} does not exist."

            cache_file = f"{cache_folder}/{filename.replace('/', '_')}_{mode}.feature_cache"
            if os.path.exists(cache_file):
                # Load feature from the cached file
                if pam == True:
                    features, features_sizes, kmp_indexs, throughputs, min_latency = pickle.load(open(cache_file, "rb"))
                else:
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
                buf_features = {}
                throughputs = {}
                min_latency = {}
                features_sizes = {}
                kmp_indexs = {}

                for task, (inputs, results) in measure_records.items():
                    if pam == True:

                        features_, buf_features_, features_size, kmp_index, normalized_throughputs, task_ids, min_latency_ =\
                            get_per_store_features_from_measure_pairs_pam(inputs, results, mode=False)
                    else:
                        features_, normalized_throughputs, task_ids, min_latency_ =\
                            get_per_store_features_from_measure_pairs(inputs, results)
                        

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
                    if pam == True:
                        buf_features[task] = buf_features_
                        features_sizes[task] = features_size
                        kmp_indexs[task] = kmp_index
                if pam == True:
                    pickle.dump((features, features_sizes, kmp_indexs, throughputs, min_latency), open(cache_file, "wb"))
                else:
                    pickle.dump((features, throughputs, min_latency), open(cache_file, "wb"))

            for task in features:
                if pam == True:
                    dataset.load_task_data(task, features[task], buf_features[task], features_sizes[task], kmp_indexs[task], throughputs[task], min_latency[task])
                else:
                    dataset.load_task_data(task, features[task], throughputs[task], min_latency[task])
            print(">> file done")
        except:
            print(">> file error, skip")

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
        if pam == True:
            del dataset.buf_features[task]
            del dataset.features_sizes[task]
            del dataset.kmp_indexs[task]

    # Save to disk
    pickle.dump(dataset, open(out_file, "wb"))

    if verbose >= 0:
        print("A dataset file is saved to %s" % out_file)

