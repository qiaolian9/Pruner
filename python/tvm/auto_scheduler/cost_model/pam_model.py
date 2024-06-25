from collections import OrderedDict
import copy
from itertools import chain
import multiprocessing
import os
import pickle
import random
import time
import io
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from einops import rearrange

logger = logging.getLogger("auto_scheduler")

from tvm.auto_scheduler.dataset import PAMDataset, LearningTask
from tvm.auto_scheduler.feature import get_per_store_features_from_states_pam
from tvm.auto_scheduler.measure_record import RecordReader
from .xgb_model import get_workload_embedding
from .cost_model import PythonBasedModel

DEFAULT_GEMM_BUFFER_SIZE = 8


class PAMDataLoader:
    def __init__(
            self,
            dataset,
            batch_size,
            device,
            use_workload_embedding=True,
            fea_norm_vec=None,
            fea_norm_vec_buf=None,
            shuffle=False,
    ):
        self.device = device
        self.shuffle = shuffle
        self.number = len(dataset)
        self.batch_size = batch_size

        self.labels = torch.empty((self.number,), dtype=torch.float32)

        self.segment_sizes_normal = torch.empty((self.number,), dtype=torch.int32)

        # Flatten features
        flatten_normal_features = []
        flatten_gemm_features = []
        
        ct = 0
        for task in dataset.features:
            throughputs = dataset.throughputs[task]
            self.labels[ct: ct + len(throughputs)] = torch.tensor(throughputs)
            task_embedding = None
            if use_workload_embedding:
                task_embedding = np.zeros(
                    (10 if use_workload_embedding else 0),
                    dtype=np.float32,
                )

                if use_workload_embedding:
                    tmp_task_embedding = get_workload_embedding(task.workload_key)
                    task_embedding[:9] = tmp_task_embedding

            # for item in dataset.features[task]:
            # step 1: extract gemm
            for id, item in enumerate(dataset.features[task]):
                self.segment_sizes_normal[ct] = item.shape[0]
                buf_fea = dataset.buf_features[task][id]

                # print(fea_item.shape)
                if task_embedding is not None:
                    tmp = np.tile(task_embedding, (item.shape[0], 1))
                    item_ = np.concatenate([item, tmp], axis=1)
                else:
                    item_ = item
                
                # print(fea_normal.shape, fea_item.shape)
                flatten_normal_features.extend(item_)
                flatten_gemm_features.append(buf_fea)
                ct += 1
        self.features_normal = torch.tensor(np.array(flatten_normal_features, dtype=np.float32))
        self.features_gemm = torch.tensor(np.array(flatten_gemm_features, dtype=np.float32))

        if fea_norm_vec is not None and fea_norm_vec_buf is not None:
            self.normalize(fea_norm_vec, fea_norm_vec_buf)

        self.feature_offsets_normal = (
                    torch.cumsum(self.segment_sizes_normal, 0, dtype=torch.int32) - self.segment_sizes_normal).cpu().numpy()
        self.iter_order = self.pointer = None

    def normalize(self, norm_vector=None, norm_vector_buf=None):
        if norm_vector is None:
            norm_vector = torch.ones((self.features_normal.shape[1],))
            for i in range(self.features_normal.shape[1]):
                max_val = self.features_normal[:, i].max().item()
                if max_val > 0:
                    norm_vector[i] = max_val

        if norm_vector_buf is None:
            norm_vector_buf = torch.ones((self.features_gemm.shape[2],))
            for i in range(self.features_gemm.shape[2]):
                max_val = self.features_gemm[..., i].max().item()
                if max_val > 0:
                    norm_vector_buf[i] = max_val

        self.features_normal /= norm_vector
        self.features_gemm /= norm_vector_buf

        return norm_vector, norm_vector_buf

    def __iter__(self):
        if self.shuffle:
            self.iter_order = torch.randperm(self.number)
        else:
            self.iter_order = torch.arange(self.number)
        self.pointer = 0

        return self

    def sample_batch(self, batch_size):
        raise NotImplemented
        batch_indices = np.random.choice(self.number, batch_size)
        return self._fetch_indices(batch_indices)

    def __next__(self):
        if self.pointer >= self.number:
            raise StopIteration

        batch_indices = self.iter_order[self.pointer: self.pointer + self.batch_size]
        self.pointer += self.batch_size
        return self._fetch_indices(batch_indices)

    def _fetch_indices(self, indices):
        segment_sizes_normal = self.segment_sizes_normal[indices]

        feature_offsets_normal = self.feature_offsets_normal[indices]
        feature_indices = np.empty((segment_sizes_normal.sum(),), dtype=np.int32)
        ct = 0
        for (offset, seg_size) in zip(feature_offsets_normal, segment_sizes_normal.numpy()):
            feature_indices[ct: ct + seg_size] = np.arange(offset, offset + seg_size, 1)
            ct += seg_size
        
        features_normal = self.features_normal[feature_indices]
        labels = self.labels[indices]
        feature_gemm_pattern = self.features_gemm[indices]
        
        return (segment_sizes_normal.to(self.device), features_normal.to(self.device), labels.to(self.device), feature_gemm_pattern.to(self.device))

    def __len__(self):
        return self.number


class PAMModule(nn.Module):  
    def __init__(self, in_dim, buf_in_dim, hidden_dim,  mha_hidden_dim, out_dim, attention_head, use_norm=True,):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.segment_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        if use_norm:
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.norm = nn.Identity()

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim + mha_hidden_dim, hidden_dim),
            # nn.Linear(mha_hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.l0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.l1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # encode for gemm
        self.gemm_encoder = nn.Sequential(
            nn.Linear(buf_in_dim, mha_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mha_hidden_dim // 2, mha_hidden_dim),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(
            mha_hidden_dim, attention_head)
        
        # self.norm_gemm = nn.LayerNorm(hidden_dim)
        
        self.decoder = nn.Linear(hidden_dim, out_dim)

    def freeze_for_fine_tuning(self):
        for x in self.segment_encoder.parameters():
            x.requires_grad_(False)

    def forward(self, segment_sizes, features, feature_gemm_pattern):
        n_seg = segment_sizes.shape[0]
        device = features.device

        segment_sizes = segment_sizes.long()

        # normal version
        features = self.segment_encoder(
            features
        )
        segment_indices = torch.repeat_interleave(
            torch.arange(n_seg, device=device), segment_sizes
        )

        n_dim = features.shape[1]
        segment_sum = torch.scatter_add(
            torch.zeros((n_seg, n_dim), dtype=features.dtype, device=device),
            0,
            segment_indices.view(-1, 1).expand(-1, n_dim),
            features,
        )

        # gemm branch
        gemm_mha_output = self.gemm_encoder(feature_gemm_pattern).transpose(0, 1)
        gemm_mha_output = self.attention(gemm_mha_output, gemm_mha_output, gemm_mha_output)[0] +  gemm_mha_output#NBD
        gemm_mha_output = gemm_mha_output.sum(0).squeeze()
        
        output = torch.cat([segment_sum, gemm_mha_output],dim=1)
        output = self.fuse(output)
        # output = segment_sum + gemm_mha_output

        # ======== ablition 1# wo gemm tiling pattern =========
        # features = self.segment_encoder(
        #     features
        # )
        # segment_indices = torch.repeat_interleave(
        #     torch.arange(n_seg, device=device), segment_sizes
        # )

        # n_dim = features.shape[1]
        # segment_sum = torch.scatter_add(
        #     torch.zeros((n_seg, n_dim), dtype=features.dtype, device=device),
        #     0,
        #     segment_indices.view(-1, 1).expand(-1, n_dim),
        #     features,
        # )
        # output = segment_sum
        # output = segment_sum + gemm_mha_output
        # # ======== ablition 2# wo stmt =========

        # # gemm branch
        # gemm_mha_output = self.gemm_encoder(feature_gemm_pattern).transpose(0, 1)
        # gemm_mha_output = self.attention(gemm_mha_output, gemm_mha_output, gemm_mha_output)[0] +  gemm_mha_output#NBD
        # # print(gemm_mha_output[1], 'after', gemm_mha_output[1].shape)
        # gemm_mha_output = gemm_mha_output.sum(0).squeeze()
        
        # # output = torch.cat([segment_sum, gemm_mha_output],dim=1)
        # output = self.fuse(output)
        # # output = segment_sum + gemm_mha_output
        # # ======== done 2# wo stmt =========

        
        # decoder
        output = self.norm(output)
        output = self.l0(output) + output
        output = self.l1(output) + output
        output = self.decoder(
            output
        ).squeeze()
        
        return output
    
    def initialize_module(self):
        print("========== init module ==========")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                print('nn.linear init')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



def moving_average(average, update):
    if average is None:
        return update
    else:
        return average * 0.95 + update * 0.05


class PAMModelInternal:
    def __init__(self, device=None, few_shot_learning="base_only", use_workload_embedding=True,
                 loss_type='lambdaRankLoss'):
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda:0'
            else:
                device = 'cpu'
        print(device)


        self.net_params = {
            "type": "PAM",
            "fea_size": DEFAULT_GEMM_BUFFER_SIZE,
            "in_dim": 164 + (10 if use_workload_embedding else 0), 
            "buf_in_dim": 23,  
            "hidden_dim": 256, # 256 - head 4; 512 - 8
            "mha_hidden_dim": 128, # 256 - head 4; 512 - 8
            "attention_head": 4, # 256 - head 4; 512 - 8
            # "dim_head": 32,
            "out_dim": 1,
        }


        self.target_id_dict = {}
        self.loss_type = loss_type
        self.n_epoch = 100
        self.lr = 1e-3
        self.iter = 0
        self.siamese_iter = 4
        

        if loss_type == 'rmse':
            self.loss_func = nn.MSELoss()
        elif loss_type == 'lambdaRankLoss':
            self.loss_func = LambdaRankLoss()
            self.lr = 7e-4
            self.n_epoch = 50
        else:
            raise ValueError("Invalid loss type: " + loss_type)

        self.grad_clip = 0.5
        self.few_shot_learning = few_shot_learning
        self.fea_norm_vec = None
        self.fea_norm_vec_buf = None
        self.use_workload_embedding = use_workload_embedding

        # Hyperparameters for self.fit_base
        # self.batch_size = 512
        self.batch_size = 512
        self.infer_batch_size = 4096
        self.wd = 1e-6
        self.device = device
        self.print_per_epoches = 5

        self.m = 0.99

        # Hyperparameters for fine-tuning
        self.fine_tune_lr = 7e-4
        self.fine_tune_batch_size = 512
        self.fine_tune_num_steps = 10
        self.fine_tune_wd = 0

        # models
        self.base_model = None
        self.siamese_model = None

    def fit_base(self, train_set, valid_set=None, valid_train_set=None, epoch=None, save_dir=None):
        self.iter += 1
        if self.few_shot_learning == 'base_only':
            self.base_model = self._fit_a_model(train_set, valid_set, valid_train_set, n_epoch=epoch, siamese_update = False, save_dir=save_dir)
        elif self.few_shot_learning == 'fine_tune_mix_task':
            self.base_model = self._fine_tune_a_model(self.base_model, train_set, valid_set, n_epoch=epoch, save_dir=save_dir)
        elif self.few_shot_learning == 'siamese_update':
            if self.iter % self.siamese_iter == 0:
                self.base_model = self._fit_a_model(train_set, valid_set, valid_train_set, n_epoch=epoch, siamese_update = True, save_dir=save_dir)

    def predict(self, dataset):
        return self._predict_a_dataset(self.base_model, dataset)
    
    def make_net(self, params, siamese_update=False):
        net =  PAMModule(params["in_dim"], params["buf_in_dim"], params["hidden_dim"], params["mha_hidden_dim"],
                        params["out_dim"], params["attention_head"]).to(self.device)
        if siamese_update == False:
            # net.initialize_module()
            pass
        else:
            print("siamese update mode")
            net.load_state_dict(self.siamese_model.state_dict(), strict=False)
        return net

    @torch.no_grad()
    def _siamese_update_key_encoder(self, net):
        """
        siamese update of the base model
        """
        print("siamese update mode done")
        for param_q, param_k in zip(
            self.siamese_model.parameters(), net.parameters()
        ):
            param_q.data = param_q.data * self.m + param_k.data * (1.0 - self.m)

    def _fit_a_model(self, train_set, valid_set=None, valid_train_set=None, n_epoch=None, siamese_update=False, save_dir=None):
        print("=" * 60 + "\nFit a net. Train size: %d" % len(train_set))

        for task in train_set.tasks():
            self.register_new_task(task)

        train_loader = PAMDataLoader(
            train_set, self.batch_size, self.device,
            shuffle=True
        )

        # Normalize features
        # if self.fea_norm_vec is None:
        #     self.fea_norm_vec, self.fea_norm_vec_buf = train_loader.normalize()
        # else:
        #     train_loader.normalize(self.fea_norm_vec, self.fea_norm_vec_buf)
        self.fea_norm_vec, self.fea_norm_vec_buf = train_loader.normalize()

        if valid_set:
            for task in valid_set.tasks():
                self.register_new_task(task)
            valid_loader = PAMDataLoader(valid_set, self.infer_batch_size, self.device,
                                              fea_norm_vec=self.fea_norm_vec, fea_norm_vec_buf=self.fea_norm_vec_buf)

        n_epoch = n_epoch or self.n_epoch

        # net = self.make_net(self.net_params, siamese_update).to(self.device)
        net = self.make_net(self.net_params, siamese_update)
        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.wd
        )
        if save_dir is not None:
            self.print_per_epoches = 1
            print(save_dir)

        if n_epoch >= 70:
            early_stop = 30
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epoch // 3, gamma=0.5)
        else:
            early_stop = n_epoch // 3
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_epoch // 3, gamma=1)

        train_loss = None
        best_epoch = None
        best_train_loss = 1e10
        for epoch in range(n_epoch):
            tic = time.time()

            # train
            net.train()
            for batch, (segment_sizes, features, labels, feature_gemm_pattern) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self.loss_func(net(segment_sizes, features, feature_gemm_pattern), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
                optimizer.step()

                train_loss = moving_average(train_loss, loss.item())
                # train_loss = loss.item()
            lr_scheduler.step()

            train_time = time.time() - tic

            if epoch % self.print_per_epoches == 0 or epoch == n_epoch - 1:

                if valid_set and valid_loader:
                    valid_loss = self._validate(net, valid_loader)
                else:
                    valid_loss = 0.0

                if self.loss_type == "rmse":
                    loss_msg = "Train RMSE: %.4f\tValid RMSE: %.4f" % (np.sqrt(train_loss), np.sqrt(valid_loss))
                elif self.loss_type in ["rankNetLoss", "lambdaRankLoss", "listNetLoss"]:
                    loss_msg = "Train Loss: %.4f\tValid Loss: %.4f" % (train_loss, valid_loss)

                # print("Epoch: %d\tBatch: %d\t%s\tTrain Speed: %.0f" % (
                #     epoch, batch, loss_msg, len(train_loader) / train_time,))
                print("Epoch: %d\tBatch: %d\t%s\tTrain Speed: %.0f\tLR: %.4e" % (
                    epoch, batch, loss_msg, len(train_loader) / train_time, optimizer.param_groups[0]['lr'],))

            # Early stop
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_epoch = epoch
                if save_dir is not None:
                    filename=f'{save_dir}/epoch={epoch}-loss={train_loss}-val_loss{valid_loss}.pkl'
                    print(filename)
                    self.saveckpt(copy.deepcopy(net), filename)
            elif epoch - best_epoch >= early_stop:
                print("Early stop. Best epoch: %d" % best_epoch)
                break

            # self.save("tmp_pam.pkl")
            
        if siamese_update:
            with torch.no_grad():  # no gradient to keys
                self._siamese_update_key_encoder(net)
        
        return net
    
    def _fine_tune_a_model(self, model, train_set, valid_set=None, verbose=1, n_epoch=None, save_dir=None):
        if verbose >= 1:
            print("=" * 60 + "\nFine-tune a PAM net. Train size: %d" % len(train_set))

        for task in train_set.tasks():
            self.register_new_task(task)

        train_loader = PAMDataLoader(
            train_set, self.batch_size, self.device,
            shuffle=True
        )

        # Normalize features
        if self.fea_norm_vec is None:
            self.fea_norm_vec, self.fea_norm_vec_buf = train_loader.normalize()
        else:
            train_loader.normalize(self.fea_norm_vec, self.fea_norm_vec_buf)
        # self.fea_norm_vec, self.fea_norm_vec_buf = train_loader.normalize()

        if valid_set:
            for task in valid_set.tasks():
                self.register_new_task(task)
            valid_loader = PAMDataLoader(valid_set, self.infer_batch_size, self.device,
                                              fea_norm_vec=self.fea_norm_vec, fea_norm_vec_buf=self.fea_norm_vec_buf)

        tic = time.time()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.fine_tune_lr, weight_decay=self.fine_tune_wd)
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.fine_tune_lr, weight_decay=self.wd)
        best_train_loss = 1e10
        fine_tune_epoch = n_epoch if n_epoch is not None else self.fine_tune_num_steps
        for epoch in range(fine_tune_epoch):
            # train
            model.train()
            train_loss = None
            for batch, (segment_sizes, features, labels, feature_gemm_pattern) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = self.loss_func(model(segment_sizes, features, feature_gemm_pattern), labels)
                loss.backward()
                optimizer.step()

                train_loss = moving_average(train_loss, loss.item())

            if verbose >= 1:
                if valid_set:
                    valid_loss = self._validate(model, valid_loader)
                else:
                    valid_loss = 0

                if self.loss_type == "rmse":
                    loss_msg = "Train RMSE: %.4f\tValid RMSE: %.4f" % (np.sqrt(train_loss), np.sqrt(valid_loss))
                elif self.loss_type in ["rankNetLoss", "lambdaRankLoss", "listNetLoss"]:
                    loss_msg = "Train Loss: %.4f\tValid Loss: %.4f" % (train_loss, valid_loss)
                print("Fine-tune step: %d\tBatch: %d\t%s\tTime: %.1f\tLR: %.4e" % (epoch, batch, loss_msg, time.time() - tic, optimizer.param_groups[0]['lr'],))

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                if save_dir is not None:
                    filename=f'{save_dir}/epoch={epoch}-loss={train_loss}-val_loss{valid_loss}.pkl'
                    print(filename)
                    self.saveckpt(copy.deepcopy(model), filename)


        return model

    def register_new_task(self, task):
        target = str(task.target)

        if target not in self.target_id_dict:
            self.target_id_dict[target] = len(self.target_id_dict)


    def _validate(self, model, valid_loader):
        model.eval()
        valid_losses = []

        for segment_sizes, features, labels, feature_gemm_pattern in valid_loader:
            preds = model(segment_sizes, features, feature_gemm_pattern)
            valid_losses.append(self.loss_func(preds, labels).item())

        return np.mean(valid_losses)

    def _predict_a_dataset(self, model, dataset):
        ret = {}
        for task in dataset.features.keys():
            features = dataset.features[task]
            buf_features = dataset.buf_features[task]
            features_sizes = dataset.features_sizes[task]
            kmp_indexs = dataset.kmp_indexs[task]
            ret[task] = self._predict_a_task(model, task, features, buf_features, features_sizes, kmp_indexs)
        return ret

    def _predict_a_task(self, model, task, features, buf_features, features_size, kmp_index):
        if model is None:
            return np.zeros(len(features), dtype=np.float32)
        tmp_set = PAMDataset.create_one_task(task, features, buf_features, features_size, kmp_index, np.zeros((len(features),)))

        preds = []
        for segment_sizes, features, labels, feature_gemm_pattern in PAMDataLoader(
                tmp_set, self.infer_batch_size, self.device,
                self.use_workload_embedding, fea_norm_vec=self.fea_norm_vec, fea_norm_vec_buf=self.fea_norm_vec_buf
        ):
            preds.append(model(segment_sizes, features, feature_gemm_pattern))
        return torch.cat(preds).detach().cpu().numpy()

    def load(self, filename):
        print(filename, os.path.isfile(filename))
        if self.device == 'cpu':
            self.base_model, _, self.fea_norm_vec, self.fea_norm_vec_buf = \
                CPU_Unpickler(open(filename, 'rb')).load()
        else:
            base_model, _, self.fea_norm_vec, self.fea_norm_vec_buf = \
                pickle.load(open(filename, 'rb'))
            if self.few_shot_learning == 'siamese_update':
                self.siamese_model = base_model.cuda() if base_model else None 
                for param_k in self.siamese_model.parameters():
                    param_k.requires_grad = False
            self.base_model = base_model.cuda() if base_model else None 
        # exit()
        # print(self.base_model)

    def save(self, filename):
        base_model = self.base_model.cpu() if self.base_model else None 
        pickle.dump((base_model,  self.few_shot_learning, self.fea_norm_vec, self.fea_norm_vec_buf),
                    open(filename, 'wb'))
        self.base_model = self.base_model.to(self.device) if self.base_model else None 

    def saveckpt(self, net, filename):
        base_model = net.cpu()
        pickle.dump((base_model,  self.few_shot_learning, self.fea_norm_vec, self.fea_norm_vec_buf),
                    open(filename, 'wb'))
        self.base_model = self.base_model.to(self.device) if self.base_model else None 

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class PAMModel(PythonBasedModel):
    """The wrapper of TransformerModelInternal. So we can use it in end-to-end search."""

    def __init__(self, few_shot_learning="base_only", disable_update=False):
        super().__init__()

        self.disable_update = disable_update
        self.model = PAMModelInternal(few_shot_learning=few_shot_learning)
        self.dataset = PAMDataset()

    def update(self, inputs, results):
        if self.disable_update or len(inputs) <= 0:
            return
        
        tic = time.time()
        self.dataset.update_from_measure_pairs(inputs, results)
        self.model.fit_base(self.dataset)
        logger.info("PAM Model Training time: %.2f s", time.time() - tic)

    def predict(self, task, states):
        # features, normalized_throughputs, task_ids, min_costs, features_size

        features, buf_features, features_size, kmp_index, _, _, _ = get_per_store_features_from_states_pam(states, task, mode=False)
        if self.model is not None:
            learning_task = LearningTask(task.workload_key, str(task.target))
            eval_dataset = PAMDataset.create_one_task(learning_task, features, buf_features, features_size, kmp_index, None)
            ret = self.model.predict(eval_dataset)[learning_task]
        else:
            ret = np.random.uniform(0, 1, (len(states),))

        # Predict 0 for invalid states that failed to be lowered.
        for idx, feature in enumerate(features):
            if feature.min() == feature.max() == 0:
                ret[idx] = float('-inf')

        return ret

    def update_from_file(self, file_name, n_lines=None):
        inputs, results = RecordReader(file_name).read_lines(n_lines)
        logger.info("TransformerModel: Loaded %s measurement records from %s", len(inputs), file_name)
        self.update(inputs, results)

    def save(self, file_name: str):
        self.model.save(file_name)

    def load(self, file_name: str):
        if self.model is None:
            self.model = PAMModelInternal()
        self.model.load(file_name)
        self.num_warmup_sample = -1


def vec_to_pairwise_prob(vec):
    s_ij = vec - vec.unsqueeze(1)
    p_ij = 1 / (torch.exp(s_ij) + 1)
    return torch.triu(p_ij, diagonal=1)


class LambdaRankLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def lamdbaRank_scheme(self, G, D, *args):
        return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(
            G[:, :, None] - G[:, None, :])

    def forward(self, preds, labels, k=None, eps=1e-10, mu=10., sigma=1., device=None):
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda:0'
            else:
                device = 'cpu'
        preds = preds[None, :]
        labels = labels[None, :]
        y_pred = preds.clone()
        y_true = labels.clone()

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)
        ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        weights = self.lamdbaRank_scheme(G, D, mu, true_sorted_by_preds)

        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs[torch.isnan(scores_diffs)] = 0.
        weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
        losses = torch.log2(weighted_probas)
        masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
        loss = -torch.sum(masked_losses)
        return loss
