import logging
from .cost_model import PythonBasedModel
from tvm.auto_scheduler.feature import get_per_store_features_from_states_psa
import numpy as np
from tvm.auto_scheduler.dataset import Dataset, LearningTask
logger = logging.getLogger("auto_scheduler")

class PSAModelInternal:
    def __init__(self, peak_performance=14900, glbmem_bandwidth=550, vec_len = 10,
                  active_blocks_per_sm = 1, sm_nums = 80, arch_sm_partition = 4, arch_warp_size = 32, reg_cap = 128):
        self.peak_performance = peak_performance
        self.glbmem_bandwidth = glbmem_bandwidth
        self.vec_len = vec_len
        self.active_blocks_per_sm = active_blocks_per_sm
        self.sm_nums = sm_nums
        self.arch_sm_partition = arch_sm_partition
        self.arch_warp_size = arch_warp_size
        self.reg_cap = reg_cap

    def predict(self, dataset):
        return self._predict_a_dataset(dataset)

    def _inf(self, psa_features):
        ret = []
        for feature in psa_features:
            total_latency = 0

            if (np.all(feature == 0)):
                ret.append(-1)
                continue

            
            grid_sizes = feature[:, -3]
            block_sizes = feature[:, -2]
            block_warps_nums_ = block_sizes / self.arch_warp_size
            block_warps_nums = np.ceil(block_warps_nums_)
            #---------------- version 6 ----------------
            # compute penalty
            # 1.warp penalty
            warp_penaltys = block_warps_nums_ / block_warps_nums
            # 2.thread penalty for reg-shared reuse
            # print(stmt_feature[6])
            thread_penaltys = 1 + 1 / feature[:, 6]
            memory_penaltys = 1
            # 3.peak compute penalty
            sche_units_nums = grid_sizes
            sche_units_totals = np.full(sche_units_nums.shape, self.sm_nums * self.active_blocks_per_sm)

            activate_warps = np.ones(grid_sizes.shape)
            midKernel = grid_sizes < self.active_blocks_per_sm * self.sm_nums
            tinykernel = np.logical_and(midKernel, block_warps_nums < self.arch_sm_partition)
            samllkernel = np.logical_and(midKernel, block_warps_nums >= self.arch_sm_partition)
            activate_warps[tinykernel] = block_warps_nums[tinykernel]
            activate_warps[samllkernel] = self.arch_sm_partition
            sche_units_nums = sche_units_nums * activate_warps
            sche_units_totals[midKernel] = self.arch_sm_partition * self.sm_nums
            peak_penaltys = sche_units_nums / (np.ceil(sche_units_nums / sche_units_totals) * sche_units_totals)
            # 4.reg_cap_penalty
            reg_penaltys = feature[:, 2] / self.reg_cap
            reg_penaltys[feature[:, 2] < self.reg_cap] = 1

            # step 1: compute latency
            # ------------------ Ablition experiment compute -------------------- #
            # default config 
            # compute_kernel_tp = self.peak_performance * peak_penalty * warp_penalty / reg_penalty
            # ablition No.1 on PSA peak_penalty
            # compute_kernel_tp = self.peak_performance * warp_penalty / reg_penalty
            # ablition No.2 on PSA warp_penalty
            # compute_kernel_tp = self.peak_performance * peak_penalty / reg_penalty
            # ablition No.3 on PSA reg_penalty
            # compute_kernel_tp = self.peak_performance * peak_penalty * warp_penalty
            # ablition No.4 on PSA thread_penalty
            # compute_kernel_tp = self.peak_performance * peak_penalty * warp_penalty / reg_penalty
            # thread_penalty = 1
            # ------------------ Ablition experiment Done -------------------- #
            compute_kernel_tp = self.peak_performance * peak_penaltys * warp_penaltys / reg_penaltys
            compute_workload_totals = feature[:, 1]
            compute_time_ns = compute_workload_totals * thread_penaltys / compute_kernel_tp / 1e6 # * 1000000 to us
            # step 2: DRAM latency
            # ------------------ Ablition experiment mem -------------------- #
            # default config 
            # memory_latency_tensor = stmt_feature[5] * memory_penalty / self.glbmem_bandwidth * (1000) / (1024 * 1024 * 1024)
            # ablition No.1 on PSA mem
            # memory_latency_tensor = 1
            # ------------------ Ablition experiment Done -------------------- #
            memory_latency_tensors = feature[:, 5] * memory_penaltys / self.glbmem_bandwidth * (1000) / (1024 * 1024 * 1024)

            total_latency += sum(np.max([compute_time_ns, memory_latency_tensors], axis=0))

            ret.append(1 / total_latency)

        return ret  

    def _predict_a_dataset(self, dataset):
        ret = {}
        for task, features in dataset.features.items():
            ret[task] = self._predict_a_task(task, features)
        return ret

    def _predict_a_task(self, task, features):
        # tmp_set = Dataset.create_one_task(task, features, np.zeros((len(features),)))

        preds = []
        preds.extend(self._inf(features))
        return np.array(preds)




class PSAModel(PythonBasedModel):
    """The wrapper of PSAModelInternal. So we can use it in end-to-end search."""
    def __init__(self, peak_performance=14900, glbmem_bandwidth=550, vec_len = 11,
                  active_blocks_per_sm = 1, sm_nums = 80, arch_sm_partition = 4, arch_warp_size = 32):
        super().__init__()
        self.peak_performance = peak_performance
        self.glbmem_bandwidth = glbmem_bandwidth
        self.vec_len = vec_len
        self.active_blocks_per_sm = active_blocks_per_sm
        self.sm_nums = sm_nums
        self.arch_sm_partition = arch_sm_partition
        self.arch_warp_size = arch_warp_size
        
        self.model = PSAModelInternal(peak_performance, glbmem_bandwidth, vec_len,
                  active_blocks_per_sm, sm_nums, arch_sm_partition, arch_warp_size)
        self.dataset = Dataset()

    def update(self, inputs, results):
        pass

    def predict(self, task, states):
        features = get_per_store_features_from_states_psa(
            states, task)
        
        if self.model is not None:
            learning_task = LearningTask(task.workload_key, str(task.target))
            eval_dataset = Dataset.create_one_task(learning_task, features, None)
            ret = self.model.predict(eval_dataset)[learning_task]
        else:
            ret = np.random.uniform(0, 1, (len(states),))
            
        # Predict 0 for invalid states that failed to be lowered.
        for idx, feature in enumerate(features):
            if feature.min() == feature.max() == 0:
                ret[idx] = float('-inf')

        return ret
