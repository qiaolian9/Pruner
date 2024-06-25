/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file auto_scheduler/feature.cc
 * \brief Feature extraction for the cost model
 */

#include <tvm/arith/analyzer.h>
#include <tvm/auto_scheduler/feature_psa.h>
#include <tvm/auto_scheduler/measure.h>
#include <tvm/auto_scheduler/measure_record.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/parallel_for.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>


#include "../runtime/thread_storage_scope.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "search_policy/utils.h"
#include "tvm/ir/expr.h"
#include "tvm/runtime/container.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/expr.h"
#include "utils.h"

namespace tvm {
// import the function from driver_api.cc
void GetBinds(const Array<te::Tensor>& args, bool compact,
              const std::unordered_map<te::Tensor, tir::Buffer>& binds,
              Map<te::Tensor, tir::Buffer>* out_binds, Array<ObjectRef>* out_arg_list);
}  // namespace tvm

namespace tvm {
namespace auto_scheduler {

using runtime::StorageScope;
using runtime::ThreadScope;

// Buffer access type
enum class BufferAccessType : int { kRead = 0, kWrite = 1, kReadWrite = 2, kUnknownRW = 3 };
// Accesses to a buffer
struct BufferAccess {
  // data reuse type
  BufferAccessType acc_type{BufferAccessType::kUnknownRW};
  // Use a two-dimensional array to store multiple multi-dimensional accesses.
  // The innermost vector stores the multi-dimensional indices of one access.
  std::vector<std::vector<PrimExpr>> indices;
};
// Data reuse type
enum class ReuseType : int { kLoopMultipleRead = 0, kSerialMultipleReadWrite = 1, kNoReuse = 2 };
using namespace tvm::tir;
using arith::Analyzer;
// using arith::ConstIntBound;

template <class T>
using BufferMap = std::unordered_map<Buffer, T, ObjectHash, ObjectEqual>;

// Feature set of a BufferStore statement
struct FeatureSetPSA {
  /***** Group 1: Computation related features within single thread *****/ 
  float float_workload_thread;                  // The number of float ops
  float ComputeWorkload;
  
  /***** Group 2: Memory related features within single thread *****/ 
  float RegUsage;           // The number of memory workload from shared mem to reg
  float MemWorkload;
  float SharedMemFootprint;           // The number of memory workload from DRAM mem to shared mem
  float DRAM_workload;
  
  float DataReuseRenalty;           // The reuse penalty for shared mem

  /***** Group 3: bind information for GPU *****/
  float is_gpu;                     // Whether it is a GPU task
  float blockIdx_len;             // The length of blockIdx
  float threadIdx_len;            // The length of threadIdx
  float vthread_len;                // The length of virtual thread
};

// Return the extent of a for loop
int64_t GetLoopExtentPSA(const ForNode* node) {
  auto pint = node->extent.as<IntImmNode>();
  if (pint != nullptr) {
    return pint->value;
  } else {
    return 1;
  }
}

class MathOpCounter : public StmtExprVisitor {
 public:
#define VisitBinary(Type, float_ct, int_ct) \
  void VisitExpr_(const Type* op) final {   \
    if (op->a.dtype().is_float()) {         \
      float_ct++;                           \
    } else {                                \
      int_ct++;                             \
    }                                       \
    StmtExprVisitor::VisitExpr_(op);        \
  }

  VisitBinary(AddNode, float_addsub, int_addsub);
  VisitBinary(SubNode, float_addsub, int_addsub);
  VisitBinary(MulNode, float_mul, int_mul);
  VisitBinary(DivNode, float_divmod, int_divmod);
  VisitBinary(ModNode, float_divmod, int_divmod);
  VisitBinary(FloorDivNode, float_divmod, int_divmod);
  VisitBinary(FloorModNode, float_divmod, int_divmod);
  VisitBinary(MaxNode, float_cmp, int_cmp);
  VisitBinary(MinNode, float_cmp, int_cmp);
  VisitBinary(EQNode, float_cmp, int_cmp);
  VisitBinary(NENode, float_cmp, int_cmp);
  VisitBinary(LTNode, float_cmp, int_cmp);
  VisitBinary(LENode, float_cmp, int_cmp);
  VisitBinary(GTNode, float_cmp, int_cmp);
  VisitBinary(GENode, float_cmp, int_cmp);

#undef VisitBinary

  void VisitExpr_(const AndNode* op) final {
    bool_op++;
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const OrNode* op) final {
    bool_op++;
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const NotNode* op) final {
    bool_op++;
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const SelectNode* op) final {
    select_op++;
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode* op) final {
    auto* pop = op->op.as<OpNode>();
    ICHECK(pop != nullptr);
    auto effect_kind = op_call_effect_[GetRef<Op>(pop)];
    bool is_pure =
        effect_kind == CallEffectKind::kPure || effect_kind == CallEffectKind::kExprAnnotation;

    if (is_pure) {
      if (op->dtype.is_float()) {
        float_math_func++;
      } else {
        int_math_func++;
      }
    } else {
      if (op->dtype.is_float()) {
        float_other_func++;
      } else {
        int_other_func++;
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  // todo(merrymercy): Detect MAD (Multiply–add)
  size_t float_mad{0};         // The number of float MAD (Multiply–add) ops
  size_t float_addsub{0};      // The number of float add and sub ops
  size_t float_mul{0};         // The number of float multiply ops
  size_t float_divmod{0};      // The number of float div and mod ops
  size_t float_cmp{0};         // The number of float comparison ops
  size_t float_math_func{0};   // The number of float math func calls
  size_t float_other_func{0};  // The number of other float func calls
  size_t int_mad{0};           // The number of integer MAD (Multiply–add) ops
  size_t int_addsub{0};        // The number of integer add and sub ops
  size_t int_mul{0};           // The number of float multiply ops
  size_t int_divmod{0};        // The number of float div and mod ops
  size_t int_cmp{0};           // The number of float comparison ops
  size_t int_math_func{0};     // The number of float math func calls
  size_t int_other_func{0};    // The number of other float func calls
  size_t bool_op{0};           // The number of bool ops
  size_t select_op{0};         // The number of select ops

  OpAttrMap<TCallEffectKind> op_call_effect_ = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
};

// Extract all buffer accesses in an expr
class BufferAccessExtractorPSA : public StmtExprVisitor {
 public:
  void ExtractReads(const PrimExpr& expr) { this->VisitExpr(expr); }

  void InsertAccess(const Buffer& buf, BufferAccessType acc_type, const Array<PrimExpr>& indices) {
    BufferAccess& acc = buf_accesses[buf];
    acc.acc_type = acc_type;
    acc.indices.push_back(std::vector<PrimExpr>(indices.begin(), indices.end()));
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    BufferAccess& acc = buf_accesses[op->buffer];
    switch (acc.acc_type) {
      case BufferAccessType::kRead:
        break;
      case BufferAccessType::kWrite:
        acc.acc_type = BufferAccessType::kReadWrite;
        break;
      case BufferAccessType::kReadWrite:
        break;
      case BufferAccessType::kUnknownRW:
      default:
        acc.acc_type = BufferAccessType::kRead;
        break;
    }
    if (acc.acc_type != BufferAccessType::kReadWrite) {
      // If a buffer is both read and written, in the tvm DSL, it must be a update,
      // so the indices should be the same. Then we can skip appending indices for it.
      // Otherwise we do the following.
      buf_accesses[op->buffer].indices.push_back(
          std::vector<PrimExpr>(op->indices.begin(), op->indices.end()));
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  BufferMap<BufferAccess> buf_accesses;
};


// Return whether a var is in an expr
bool VarInExprPSA(const Var& var, const PrimExpr& expr) {
  bool find = false;

  PostOrderVisit(expr, [&find, &var](const ObjectRef& node) {
    if (find) {
      return;
    }

    if (const VarNode* op = node.as<VarNode>()) {
      if (op == var.get()) {
        find = true;
      }
    }
  });

  return find;
}


// Find SAxis
void FindSAxisPSA(
    const Buffer& buf, const std::vector<std::vector<PrimExpr>>& indices,
    const std::vector<const ForNode*>& for_loop_stack, std::vector<const ForNode*>& SAxis) {
  // // std::cout << "ComputeReusePSA" << std::endl;
  for (int i = static_cast<int>(for_loop_stack.size()) - 1; i >= 0; --i) {
    const ForNode* cur_for = for_loop_stack[i];
    // // std::cout << "loop idx\t#" << i << "\tname_hint\t" <<cur_for->loop_var->name_hint << std::endl;
    if (cur_for->loop_var->name_hint == "threadIdx.x" || cur_for->loop_var->name_hint == "blockIdx.x") {
      continue;
    }
    bool find = false;

    for (size_t j = 0; j < indices.size(); j++) {
      for (size_t k = 0; k < indices[j].size(); k++) {
        // // std::cout << indices[j][k] << std::endl;
        if (VarInExprPSA(cur_for->loop_var, indices[j][k])) {
          find = true;
          break;
        }
      }
      if (find) {
        SAxis.push_back(cur_for);
        break;
      }
    }
  }
}

float MemWorkloadPSA(
    const std::vector<std::vector<PrimExpr>>& indices, std::vector<const ForNode*>& for_loop_stack) {
  // // std::cout << "ComputeReusePSA" << std::endl;
  float reuse_ct = 1.0f;
  for (int i = static_cast<int>(for_loop_stack.size()) - 1; i >= 0; --i) {
    const ForNode* cur_for = for_loop_stack[i];
    // // std::cout << "loop idx\t#" << i << "\tname_hint\t" <<cur_for->loop_var->name_hint << std::endl;
    if (cur_for->loop_var->name_hint == "threadIdx.x" || cur_for->loop_var->name_hint == "blockIdx.x") {
      continue;
    }
    
    int64_t extent = GetLoopExtentPSA(for_loop_stack[i]);
    bool find = false;

    for (size_t j = 0; j < indices.size(); j++) {
      for (size_t k = 0; k < indices[j].size(); k++) {
        // // std::cout << indices[j][k] << std::endl;
        if (VarInExprPSA(cur_for->loop_var, indices[j][k])) {
          find = true;
          break;
        }
      }
      if (find) {
        reuse_ct *= extent;
        break;
      }
    }
  }
  return reuse_ct;
}

// Extract features for every BufferStore statement
class PerStoreFeatureExtractorPSA : public StmtExprVisitor {
 public:
//  explicit PerStoreFeatureExtractorPSA(int cache_line_size) : cache_line_size_(cache_line_size) {}

  void VisitStmt_(const AttrStmtNode* node) final {
    // // std::cout << "AttrStmtNode 0 ok" << std::endl;
    if (node->attr_key == tir::attr::thread_extent || node->attr_key == tir::attr::virtual_thread) {
      const Var& var = node->node.as<IterVarNode>()->var;
      int extent = GetIntImm(node->value);

      int* plen = nullptr;

      const std::string& name = var.get()->name_hint;
      if (node->attr_key == tir::attr::thread_extent) {
        if (name == "blockIdx.x") {
          plen = &blockIdx_x_len_;
        } else if (name == "blockIdx.y") {
          plen = &block_idx_y_len_;
        } else if (name == "blockIdx.z") {
          plen = &block_idx_z_len_;
        } else if (name == "threadIdx.x") {
          plen = &threadIdx_x_len_;
        } else if (name == "threadIdx.y") {
          plen = &thread_idx_y_len_;
        } else if (name == "threadIdx.z") {
          plen = &thread_idx_z_len_;
        } else {
          LOG(FATAL) << "invalid thread itervar " + name;
        }
      } else {
        plen = &vthread_len_;
      }

      int extent_before = *plen;
      if (node->attr_key == tir::attr::thread_extent) {
        *plen = extent;
      } else {
        *plen *= extent;
      }

      is_gpu_ = true;

      // make a fake for node for blockIdx.x or threadIdx.x
      Stmt fake_for_node = For(var, 0, extent, ForKind::kParallel, node->body);

      if (node->attr_key != tir::attr::thread_extent) {
        inner_loop_prod_ *= extent;
      } else {
        IterVar iv = Downcast<IterVar>(node->node);
        ThreadScope ts = ThreadScope::Create(iv->thread_tag);
        curr_thread_scope_.push_back(ts);
      }
      // outer_loop_prod_ *= extent;
      for_loop_stack_.push_back(fake_for_node.as<ForNode>());
      StmtExprVisitor::VisitStmt_(node);
      for_loop_stack_.pop_back();
      if (node->attr_key != tir::attr::thread_extent) {
        inner_loop_prod_ /= extent;
      } else {
        curr_thread_scope_.pop_back();
      }
      // outer_loop_prod_ /= extent;

      *plen = extent_before;
    } else if (node->attr_key == "pragma_auto_unroll_max_step") {
      int value = GetIntImm(node->value);

      int16_t old_value = cur_auto_unroll_max_step_;
      cur_auto_unroll_max_step_ = value;
      StmtExprVisitor::VisitStmt_(node);
      cur_auto_unroll_max_step_ = old_value;
    } else if (node->attr_key == tir::attr::realize_scope) {
      storage_scope_[node->node.get()] = node->value.as<StringImmNode>()->value;
      StmtExprVisitor::VisitStmt_(node);
    } else {
      StmtExprVisitor::VisitStmt_(node);
    }
  }

  void VisitStmt_(const ForNode* node) final {
    // // std::cout << "ForNode 0 ok" << std::endl;
    int64_t loop_extent = GetLoopExtentPSA(node);

    if (node->kind == ForKind::kVectorized) {
      vec_for_stack_.push_back(node);
    } else if (node->kind == ForKind::kUnrolled) {
      unroll_for_stack_.push_back(node);
    } else if (node->kind == ForKind::kParallel) {
      parallel_for_stack_.push_back(node);
    }

    // outer_loop_prod_ *= loop_extent;
    inner_loop_prod_ *= loop_extent;
    for_loop_stack_.push_back(node);
    StmtExprVisitor::VisitStmt_(node);
    for_loop_stack_.pop_back();
    // outer_loop_prod_ /= loop_extent;
    inner_loop_prod_ /= loop_extent;

    if (node->kind == ForKind::kVectorized) {
      vec_for_stack_.pop_back();
    } else if (node->kind == ForKind::kUnrolled) {
      unroll_for_stack_.pop_back();
    } else if (node->kind == ForKind::kParallel) {
      parallel_for_stack_.pop_back();
    }
  }

  void VisitStmt_(const BufferStoreNode* node) final {
    // // std::cout << "MathOpCounter\tstart" << std::endl;
    MathOpCounter math_op_counter;
    // // std::cout << "math_op_counter\tstart" << std::endl;
    math_op_counter(node->value);

    // std::cout << "BufferStoreNode\tstart" << std::endl;
    // Group 1: Computation related features
    ExtractComputationFeature(node, math_op_counter);
    // // std::cout << "group 1 ok" << std::endl;
    // Group 2: Buffer access related features (per buffer)
    // std::cout << "ExtractComputationFeature\tdone" << std::endl;
    ExtractBufferAccessFeature(node);
    // // std::cout << "group 2 ok" << std::endl;
    // std::cout << "ExtractBufferAccessFeature\tdone" << std::endl;
  }

  void VisitStmt_(const BufferRealizeNode* node) final {
    // std::cout << "BufferRealizeNode 0 ok" << std::endl;
    const auto& key = node->buffer;
    if (!buf_map_.count(key)) {
      // deduce current storage scope.
      auto it = storage_scope_.find(node->buffer.get());
      ICHECK(it != storage_scope_.end()) << "Cannot find storage scope of " << node->buffer;
      StorageScope skey;
      const std::string& strkey = it->second;
      if (strkey.length() == 0) {
        if (curr_thread_scope_.size() != 0) {
          skey.rank = runtime::DefaultStorageRank(curr_thread_scope_.back().rank);
        }
      } else {
        skey = StorageScope::Create(strkey);
      }
      buf_map_[key] = skey.to_string();
    }

    StmtExprVisitor::VisitStmt_(node);
  }

  // Extract computation related features (group 1)
  void ExtractComputationFeature(const BufferStoreNode* node,
                                 const MathOpCounter& math_op_counter){
    FeatureSetPSA& fea = buffer_features[node->buffer];
    // GPU threads
    fea.is_gpu = is_gpu_;
    fea.blockIdx_len = blockIdx_x_len_ * block_idx_y_len_ * block_idx_z_len_;
    fea.threadIdx_len = threadIdx_x_len_ * thread_idx_y_len_ * thread_idx_z_len_;
    fea.vthread_len = vthread_len_;

    // Computation related features
    fea.float_workload_thread = inner_loop_prod_ * (math_op_counter.float_mad
                    + math_op_counter.float_addsub
                    + math_op_counter.float_mul
                    + math_op_counter.float_divmod);
    fea.ComputeWorkload = fea.float_workload_thread * fea.blockIdx_len * fea.threadIdx_len;
    // // std::cout << "thread float_workload\t" <<fea.float_workload_thread << "\nComputeWorkload\t" << fea.ComputeWorkload << "\nGPU info\t" << fea.blockIdx_len << "\t#" << fea.threadIdx_len << "\t#" << fea.vthread_len << std::endl;
  }

  // Extract buffer access related features (group 2)
  void ExtractBufferAccessFeature(const BufferStoreNode* node) {

  // float RegUsage;     m*n + m + n      // The number of memory workload from shared mem to reg
  // float MemWorkload;  m*n + mK + Kn
  // float SharedMemFootprint;           // The number of memory workload from DRAM mem to shared mem
  // float DRAM_workload;


    FeatureSetPSA& fea = buffer_features[node->buffer];

    // Extract all buffer accesses
    BufferAccessExtractorPSA buf_extractor;
    buf_extractor.InsertAccess(node->buffer, BufferAccessType::kWrite, node->indices);
    buf_extractor.ExtractReads(node->value);

    // Compute touched region for all outer loops
    for (auto x : for_loop_stack_) {
      ana_.Bind(x->loop_var, Range::FromMinExtent(x->min, 1), true);
    }
    std::vector<const ForNode*> SAxis;
    FindSAxisPSA(node->buffer, buf_extractor.buf_accesses[node->buffer].indices, for_loop_stack_, SAxis);

    std::vector<float> RegUsage, SharedMemFootprint, DRAM_workload;
    fea.RegUsage = 0;
    fea.SharedMemFootprint = 0;
    fea.DRAM_workload = 0;
    fea.DataReuseRenalty = 1;
    for (const auto& x : buf_extractor.buf_accesses) {
      // const Buffer& t = x.first;
      // // std::cout << "buffer accesses\t" << t->name << std::endl;
      const BufferAccess& acc = x.second;
      // for i in 2:
      //   for j in 2:
      //     for k in 1024:
      //  i : 2, j : 2, C = 2 * 2 =4 
      //  A : 2 * 1024
      //       C[i, j]  = A [i, k]


      if (fea.float_workload_thread != 0) {
        fea.RegUsage += MemWorkloadPSA(acc.indices, SAxis);
        fea.MemWorkload += MemWorkloadPSA(acc.indices, for_loop_stack_);
      } else if (buf_map_.count(node->buffer) && StrStartsWith(buf_map_[node->buffer], "shared")){
        fea.SharedMemFootprint += MemWorkloadPSA(acc.indices, SAxis) * fea.threadIdx_len;
        fea.DRAM_workload += fea.SharedMemFootprint * fea.blockIdx_len;
        break;
      } else {
        fea.DRAM_workload += inner_loop_prod_ * fea.blockIdx_len * fea.threadIdx_len;
        break;
      }
    }
    if (fea.float_workload_thread != 0 && fea.MemWorkload != 0 ) {
      fea.DataReuseRenalty += fea.float_workload_thread / fea.MemWorkload;
    }
  }

  // Stores FeatureSet for every buffer
  BufferMap<FeatureSetPSA> buffer_features;

 private:
  // The shared arithmetic analyzer
  Analyzer ana_;

  // // The product of outer loop
  // float outer_loop_prod_ = 1.0f;
  // The product of inner loop within single thread
  float inner_loop_prod_ = 1.0f;

  // The stacks to store parent loops during DFS
  std::vector<const ForNode*> for_loop_stack_;
  std::vector<const ForNode*> parallel_for_stack_;
  std::vector<const ForNode*> vec_for_stack_;
  std::vector<const ForNode*> unroll_for_stack_;
  // Storage scope
  std::unordered_map<const Object*, std::string> storage_scope_;
  // Buffer map
  std::unordered_map<Buffer, std::string, ObjectPtrHash, ObjectPtrEqual> buf_map_;
  // The current thread scope.
  std::vector<ThreadScope> curr_thread_scope_;

  // GPU-related features
  bool is_gpu_{false};
  int blockIdx_x_len_{1};
  int block_idx_y_len_{1};
  int block_idx_z_len_{1};
  int threadIdx_x_len_{1};
  int thread_idx_y_len_{1};
  int thread_idx_z_len_{1};
  int vthread_len_{1};
  int16_t cur_auto_unroll_max_step_{0};
  std::unordered_map<const ForNode*,
                     BufferMap<std::vector<std::tuple<BufferAccessType, int64_t, int>>>>
      for_touch_regions_;

  // const int cache_line_size_ = 64;
};

void GetPerStoreFeaturePSA(const Stmt& stmt, 
                        std::vector<float>* ret) {
  // std::cout << "GetPerStoreFeaturePSA\t func in start" << std::endl;
  PerStoreFeatureExtractorPSA extractor;
  // std::cout << "GetPerStoreFeaturePSA\t func in start 1" << std::endl;
  extractor(stmt);

  // 5 * (com + mem)
  // std::cout << "GetPerStoreFeaturePSA\t func in start 2" << std::endl;

  ret->push_back(extractor.buffer_features.size());
  // if (extractor.buffer_features.size() == 0) {
  //   // std::cout << stmt << std::endl;
  // }
  for (const auto& x : extractor.buffer_features) {
    const FeatureSetPSA& fea_set = x.second;

    /***** Group 1: Computation related features within single thread *****/
    ret->push_back(fea_set.float_workload_thread);
    ret->push_back(fea_set.ComputeWorkload);

    /***** Group 2: memory related features within single thread *****/
    ret->push_back(fea_set.RegUsage);
    ret->push_back(fea_set.MemWorkload);
    ret->push_back(fea_set.SharedMemFootprint);
    ret->push_back(fea_set.DRAM_workload);

    ret->push_back(fea_set.DataReuseRenalty);

    /***** Group 3: bind info *****/
    ret->push_back(fea_set.is_gpu);
    ret->push_back(fea_set.blockIdx_len);
    ret->push_back(fea_set.threadIdx_len);
    ret->push_back(fea_set.vthread_len);
  }
  // std::cout << "GetPerStoreFeaturePSA\t func in done" << std::endl;
}

// void GetPerStoreFeaturesWorkerFuncPSA(const SearchTask& task, const State& state, int max_n_bufs,
//                                    std::vector<float>* feature, std::atomic<int>* error_ct) {
void GetPerStoreFeaturesWorkerFuncPSA(const SearchTask& task, const State& state,
                                   std::vector<float>* feature, std::atomic<int>* error_ct) {
  te::Schedule sch;
  Array<te::Tensor> tensors;

  // NOTE: Currently, feature extraction with and without layout rewrite
  // returns the same feature vector, so we do not turn on layout rewrite here.
  // In the future, we can improve the feature extraction to reflect this difference.
  std::tie(sch, tensors) = task->compute_dag.ApplySteps(state->transform_steps);
  sch = sch.normalize_for_feature_extraction();
  auto bounds = te::InferBound(sch);

  try {
    auto stmt = te::ScheduleOps(sch, bounds, false);
    Map<te::Tensor, te::Buffer> out_binds;
    Array<ObjectRef> out_arg_list;
    bool compact = te::VerifyCompactBuffer(stmt);
    const std::string& name = "main";
    GlobalVar global_var(name);

    // Copied from driver_api.cc::lower
    auto pass_ctx = tvm::transform::PassContext::Current();
    GetBinds(tensors, compact, std::unordered_map<te::Tensor, te::Buffer>(), &out_binds,
             &out_arg_list);
    tir::PrimFunc f = te::SchedulePostProcToPrimFunc(out_arg_list, std::move(stmt), out_binds);
    f = WithAttr(std::move(f), "global_symbol", runtime::String(name));

    bool noalias = pass_ctx->GetConfig<Bool>("tir.noalias", Bool(true)).value();
    bool disable_vectorize =
        pass_ctx->GetConfig<Bool>("tir.disable_vectorize", Bool(false)).value();
    bool instrument_bound_checkers =
        pass_ctx->GetConfig<Bool>("tir.instrument_bound_checkers", Bool(false)).value();

    if (noalias) {
      f = WithAttr(std::move(f), "tir.noalias", Bool(true));
    }
    auto mod = IRModule(Map<GlobalVar, BaseFunc>({{global_var, f}}));

    if (IsGPUTask(task)) {
      auto pass_list = Array<tvm::transform::Pass>();
      // Phase 0
      pass_list.push_back(tir::transform::InjectPrefetch());
      pass_list.push_back(tir::transform::StorageFlatten(64, instrument_bound_checkers));
      // Phase 1
      pass_list.push_back(tir::transform::NarrowDataType(32));
      pass_list.push_back(tir::transform::Simplify());
      pass_list.push_back(tir::transform::VectorizeLoop(!disable_vectorize));
      pass_list.push_back(tir::transform::InjectVirtualThread());
      pass_list.push_back(tir::transform::StorageRewrite());
      pass_list.push_back(tir::transform::Simplify());
      tvm::Map<String, tvm::PrimExpr> gpu_params{
          {"max_shared_memory_per_block", task->hardware_params->max_shared_memory_per_block},
          {"max_local_memory_per_block", task->hardware_params->max_local_memory_per_block},
          {"max_threads_per_block", task->hardware_params->max_threads_per_block},
          {"max_vector_bytes", task->hardware_params->vector_unit_bytes},
          {"max_vthread", task->hardware_params->max_vthread_extent},
      };
      pass_list.push_back(tir::transform::VerifyGPUCode(gpu_params));
      const auto& optimize = tir::transform::Sequential(pass_list);
      optimize(mod);
    }
    const auto& optimize =
        tir::transform::Sequential(Array<tvm::transform::Pass>{tir::transform::Simplify()});
    mod = optimize(std::move(mod));
    const auto& it = mod->functions.find(global_var);
    ICHECK(it != mod->functions.end());
    const auto& prim_func = (*it).second.as<PrimFuncNode>();
    // GetPerStoreFeaturePSA(prim_func->body, task->hardware_params->cache_line_bytes, max_n_bufs,
    //                    feature);
    // // std::cout << "prim_func.body\n" << prim_func->body << std::endl;
    // std::cout << "GetPerStoreFeaturePSA\tstart" << std::endl;
    GetPerStoreFeaturePSA(prim_func->body, feature);
    // std::cout << "GetPerStoreFeaturePSA\tdone" << std::endl;
  } catch (Error& e) {
    (*error_ct)++;
  }
}

// void GetPerStoreFeaturesFromStatesPSA(const Array<State>& states, const SearchTask& task,
//                                    int skip_first_n_feature_extraction, int max_n_bufs,
//                                    std::vector<std::vector<float>>* features) {
//   // extract features
//   features->assign(states.size(), std::vector<float>());

//   std::atomic<int> error_ct(0);

//   support::parallel_for(skip_first_n_feature_extraction, states.size(),
//                         [&task, &states, &max_n_bufs, &features, &error_ct](int i) {
//                           GetPerStoreFeaturesWorkerFuncPSA(task, states[i], max_n_bufs,
//                                                         &(*features)[i], &error_ct);
//                         });
// }
void GetPerStoreFeaturesFromStatesPSA(const Array<State>& states, const SearchTask& task,
                                   int skip_first_n_feature_extraction, 
                                   std::vector<std::vector<float>>* features) {
  // extract features
  features->assign(states.size(), std::vector<float>());

  std::atomic<int> error_ct(0);

  support::parallel_for(skip_first_n_feature_extraction, states.size(),
                        [&task, &states, &features, &error_ct](int i) {
                          GetPerStoreFeaturesWorkerFuncPSA(task, states[i],
                                    &(*features)[i], &error_ct);
                        });
}

void GetPerStoreFeaturesFromStatesPSA(const Array<State>& states, const std::vector<SearchTask>& tasks,
                                   int skip_first_n_feature_extraction, 
                                   std::vector<std::vector<float>>* features) {
  // extract features
  features->assign(states.size(), std::vector<float>());

  std::atomic<int> error_ct(0);

  support::parallel_for(skip_first_n_feature_extraction, states.size(),
                        [&tasks, &states, &features, &error_ct](int i) {
                          GetPerStoreFeaturesWorkerFuncPSA(tasks[i], states[i],
                                    &(*features)[i], &error_ct);
                        });
}

void GetPerStoreFeaturesFromMeasurePairsPSA(const Array<MeasureInput>& inputs,
                                         const Array<MeasureResult>& results,
                                         int skip_first_n_feature_extraction,
                                         std::vector<std::vector<float>>* features,
                                         std::vector<float>* normalized_throughputs,
                                         std::vector<int>* task_ids,
                                         std::vector<float>* min_costs) {
  Array<State> states;
  std::vector<SearchTask> tasks;

  normalized_throughputs->clear();
  task_ids->clear();
  min_costs->clear();

  // (workload_key, target) -> (search_task, task_id)
  std::unordered_map<std::pair<std::string, std::string>, std::pair<SearchTask, size_t>> task_cache;

  const auto* workload_key_to_tensors =
      tvm::runtime::Registry::Get("auto_scheduler.workload_key_to_tensors");
  ICHECK(workload_key_to_tensors != nullptr);

  tasks.reserve(inputs.size());
  normalized_throughputs->reserve(inputs.size());
  task_ids->reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    float cost = static_cast<float>(FloatArrayMean(results[i]->costs));
    const std::string& workload_key = inputs[i]->task->workload_key;
    SearchTask task;

    size_t task_id;
    std::pair<std::string, std::string> key(workload_key, inputs[i]->task->target->str());
    auto find_res = task_cache.find(key);
    if (find_res == task_cache.end()) {
      if (inputs[i]->task->compute_dag.defined()) {  // the measure input is complete
        task = inputs[i]->task;
      } else {
        // The measure input is incomplete, rebuild task for incomplete measure pairs read from file
        try {
          Array<te::Tensor> tensors = (*workload_key_to_tensors)(workload_key);
          task =
              SearchTask(ComputeDAG(tensors), workload_key, inputs[i]->task->target,
                         inputs[i]->task->target_host, inputs[i]->task->hardware_params,
                         inputs[i]->task->layout_rewrite_option, inputs[i]->task->task_input_names);
        } catch (std::exception& e) {
          // Cannot build ComputeDAG from workload key, the task may have not been registered in
          // this search round
          continue;
        }
      }
      task_id = task_cache.size();

      // compute min cost for each task
      task_cache.insert(std::make_pair(key, std::make_pair(task, task_id)));
      min_costs->push_back(cost);
    } else {
      std::tie(task, task_id) = find_res->second;
      (*min_costs)[task_id] = std::min((*min_costs)[task_id], cost);
    }

    tasks.push_back(std::move(task));
    task_ids->push_back(task_id);
    states.push_back(inputs[i]->state);
    normalized_throughputs->push_back(cost);
  }

  for (size_t i = 0; i < normalized_throughputs->size(); ++i) {
    (*normalized_throughputs)[i] = (*min_costs)[(*task_ids)[i]] / (*normalized_throughputs)[i];
  }
  // std::cout << "GetPerStoreFeaturesFromStatesPSA\t start" << std::endl;
  GetPerStoreFeaturesFromStatesPSA(states, tasks, skip_first_n_feature_extraction, 
                                features);
  // std::cout << "GetPerStoreFeaturesFromStatesPSA\t done" << std::endl;
}

TVMByteArray SerializeFeaturesPSA(std::vector<std::vector<float>>&& features,
                               std::vector<float>&& normalized_throughputs,
                               std::vector<int>&& task_ids,
                               std::vector<float>&& min_costs,
                               std::vector<char>* out_data) {
  size_t total_bytes = 0;
  std::vector<int> size_vector;

  int n = features.size();

  // kernel: 5 * 10 items

  // serialize sizes
  size_t size_vector_size = 1 + n + 3;
  total_bytes += size_vector_size * sizeof(int);

  size_vector.reserve(size_vector_size);
  size_vector.push_back(features.size());
  for (const auto& x : features) {
    size_vector.push_back(static_cast<int>(x.size()));
    total_bytes += sizeof(float) * x.size();
  }
  size_vector.push_back(static_cast<int>(normalized_throughputs.size()));
  total_bytes += sizeof(float) * normalized_throughputs.size();
  size_vector.push_back(static_cast<int>(task_ids.size()));
  total_bytes += sizeof(int) * task_ids.size();
  size_vector.push_back(static_cast<int>(min_costs.size()));
  total_bytes += sizeof(float) * min_costs.size();

  CHECK_EQ(size_vector.size(), size_vector_size);

  // allocate memory
  out_data->reserve(total_bytes);
  char* ptr = out_data->data();

  // serialize size_vector
  memmove(ptr, reinterpret_cast<char*>(size_vector.data()), size_vector.size() * sizeof(int));
  ptr += size_vector.size() * sizeof(int);

  // serialize features
  for (auto& x : features) {
    memmove(ptr, x.data(), sizeof(float) * x.size());
    ptr += sizeof(float) * x.size();
    x.clear();
  }

  // serialize normalized_throughputs
  memmove(ptr, reinterpret_cast<char*>(normalized_throughputs.data()),
          normalized_throughputs.size() * sizeof(int));
  ptr += normalized_throughputs.size() * sizeof(int);

  // serialize task_ids
  memmove(ptr, reinterpret_cast<char*>(task_ids.data()), task_ids.size() * sizeof(int));
  ptr += task_ids.size() * sizeof(int);

  // serialize min_costs
  memmove(ptr, reinterpret_cast<char*>(min_costs.data()), min_costs.size() * sizeof(float));
  ptr += min_costs.size() * sizeof(float);

  CHECK_EQ(ptr - out_data->data(), total_bytes);

  return TVMByteArray{out_data->data(), total_bytes};
}

// TVMByteArray SerializeFeaturesPSA(std::vector<std::vector<float>>&& features,
//                                std::vector<char>* out_data) {
//   size_t total_bytes = 0;
//   std::vector<int> size_vector;

//   int n = features.size();

//   // serialize sizes
//   size_t size_vector_size = 1 + n;
//   total_bytes += size_vector_size * sizeof(int);

//   size_vector.reserve(size_vector_size);
//   size_vector.push_back(features.size());
//   for (const auto& x : features) {
//     size_vector.push_back(static_cast<int>(x.size()));
//     total_bytes += sizeof(float) * x.size();
//   }

//   CHECK_EQ(size_vector.size(), size_vector_size);

//   // allocate memory
//   out_data->reserve(total_bytes);
//   char* ptr = out_data->data();

//   // serialize size_vector
//   memmove(ptr, reinterpret_cast<char*>(size_vector.data()), size_vector.size() * sizeof(int));
//   ptr += size_vector.size() * sizeof(int);

//   // serialize features
//   for (auto& x : features) {
//     memmove(ptr, x.data(), sizeof(float) * x.size());
//     ptr += sizeof(float) * x.size();
//     x.clear();
//   }

//   CHECK_EQ(ptr - out_data->data(), total_bytes);

//   return TVMByteArray{out_data->data(), total_bytes};
// }

TVM_REGISTER_GLOBAL("auto_scheduler.GetPerStoreFeaturesFromStatesPSA")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      Array<State> states = args[0];
      SearchTask task = args[1];

      std::vector<std::vector<float>> features;
      std::vector<float> normalized_throughputs;
      std::vector<int> task_ids;
      std::vector<float> min_costs;

      GetPerStoreFeaturesFromStatesPSA(states, task, 0, &features);
      
      
      std::vector<char> byte_data;
      *ret = SerializeFeaturesPSA(std::move(features), std::move(normalized_throughputs),
                               std::move(task_ids), std::move(min_costs), &byte_data);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.GetPerStoreFeaturesFromMeasurePairsPSA")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      Array<MeasureInput> inputs = args[0];
      Array<MeasureResult> results = args[1];
      int skip_first_n_feature_extraction = args[2];

      std::vector<std::vector<float>> features;
      std::vector<float> normalized_throughputs;
      std::vector<int> task_ids;
      std::vector<float> min_costs;

      GetPerStoreFeaturesFromMeasurePairsPSA(inputs, results, skip_first_n_feature_extraction,
                                          &features, &normalized_throughputs,
                                          &task_ids, &min_costs);

      std::vector<char> byte_data;
      *ret = SerializeFeaturesPSA(std::move(features), std::move(normalized_throughputs),
                               std::move(task_ids), std::move(min_costs), &byte_data);
    });

}  // namespace auto_scheduler
}  // namespace tvm
