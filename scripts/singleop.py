from tvm import te, auto_scheduler
import tvm
from tvm.auto_scheduler.task_scheduler import PrintTableInfo, TaskSchedulerCallback
import os
import time
import argparse
import test_config

class LogEstimatedLatency_single(TaskSchedulerCallback):
    """Log the estimated latency to the file after tuning a task.

    Parameters
    ----------
    log_file: str
        The log file path.
    """

    def __init__(self, log_file):
        if os.path.exists(log_file):  # Remove existing log
            os.remove(log_file)

        self.log_file = log_file

    def post_tune(self, task_scheduler):
        if all(cost < 1e9 for cost in task_scheduler.best_costs):
            total_latency_str = "%.3f" % (task_scheduler.cur_score * 1e3)
        else:
            total_latency_str = "N/A"
        speed_str = (
            "%.2f"
            % (task_scheduler.tasks[0].compute_dag.flop_ct / task_scheduler.best_costs[0] / 1e9)
            if task_scheduler.best_costs[0] < 1e9
            else "-"
        )
        with open(self.log_file, "a") as filep:
            filep.write(
                "ElapsedTime(s)\t%.0f\tEstimatedLatency(ms)\t%s\tSpeed(GFLOPS)\t%s\tTrials\t%d\n"
                % (
                    time.time() - task_scheduler.tic,
                    total_latency_str,
                    speed_str,
                    task_scheduler.ct,
                )
            )
            filep.flush()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=str, required=True)
    # Search task related arguments
    parser.add_argument("--op", type=str, required=True)
    parser.add_argument("--target", type=str, default='cuda -model=a100')
    parser.add_argument("--target-host", type=str, default=None)
    
    # Search strategy related arguments
    parser.add_argument("--cost-model", type=str, choices=['xgb', 'lgbm', 'random', 'xgb-no-update', 'lgbm-no-update', 'mlp', 'mlp-no-update', 'pam', 'pam-no-update',  'pam-siamese-update', 'tab', 'tab-no-update'],
                        default='mlp', help="The type of program cost model")
    parser.add_argument("--load-model", type=str, help="use pre model")
    parser.add_argument("--psa_model_type", type=str, choices=['titanv', 'orin', 'a100_40', 'a100', 'k80', 't4'],
                        default='a100_40', help="use psa model")
    args = parser.parse_args()
    target = tvm.target.Target(args.target)

    shape = [int(i) for i in args.shape.split('_')]
    op_expr = getattr(test_config, args.op)
    out = op_expr(shape)

    task = auto_scheduler.SearchTask(func=op_expr, args=(shape, "float32"), target=target)
    print("Computational DAG:")
    print(task.compute_dag)

    log_file = "{}.json".format(args.op)
    if os.path.exists(log_file):
        os.system("rm -rf %s" % log_file)

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=100, timeout=25)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=800,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        num_measures_per_round=10,
    )


    policy = auto_scheduler.search_policy.SketchPolicy(task)
    policy.generate_sketches(True)
    print("*" * 100)
    
    callback = ([PrintTableInfo(), LogEstimatedLatency_single("{}.tsv".format(args.op))])
    if args.load_model is not None:
        print(">> load file pass!")
        print(os.path.isfile(args.load_model))
    tuner = auto_scheduler.TaskScheduler([task], load_log_file=log_file, 
                                        load_model_file=args.load_model, callbacks=callback)
    policy = 'sketch.%s' % (args.cost_model)
    print(policy)
    tuner.tune(tune_option, search_policy=policy, psa_model_type=args.psa_model_type)


    del measure_ctx
