import contextlib
import os

import colossalai
import colossalai.utils as utils
import torch
import torch.nn as nn
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai import nn as col_nn
from colossalai.engine.schedule import (InterleavedPipelineSchedule, PipelineSchedule)
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn import LinearWarmupLR
from colossalai.trainer import Trainer, hooks
from colossalai.utils import is_using_pp, colo_set_process_memory_fraction
from colossalai.utils.timer import MultiTimer
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.pipeline.pipelinable import PipelinableContext
from titans.loss.lm_loss import GPTLMLoss
# from titans.model.gpt import GPTLMLoss
import subprocess

from dataset.webtext import WebtextDataset

from contextlib import contextmanager

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

def calc_local_model_size(model: torch.nn.Module):
    numel_per_device = 0
    for p in model.parameters():
        numel_per_device += p.numel()
    return numel_per_device


def main():
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    args = parser.parse_args()
    disable_existing_loggers()
    node_list = os.environ['SLURM_NODELIST']
    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    args.local_rank = int(os.environ['SLURM_LOCALID'])
    args.rank = int(os.environ['SLURM_PROCID'])
    args.host = addr

    print(args)
    # import pdb
    # pdb.set_trace()
    if args.from_torch:
        colossalai.launch_from_torch(config=args.config)
    else:
        colossalai.launch_from_slurm(config=args.config, host=args.host, port=args.port, seed=42, backend='nccl')

    logger = get_dist_logger()

    logger.info('Build data loader', ranks=[0])
    test_ds = WebtextDataset(os.environ['DATA'], seq_len=gpc.config.SEQ_LEN)
    test_dataloader = utils.get_dataloader(test_ds,
                                            seed=42,
                                            batch_size=gpc.config.BATCH_SIZE,
                                            pin_memory=True,
                                            shuffle=True,
                                            drop_last=True)

    logger.info('Build model', ranks=[0])
    use_pipeline = is_using_pp()
    use_interleaved = hasattr(gpc.config.model, 'num_chunks')
    num_chunks = getattr(gpc.config.model, 'num_chunks', 1)
    use_zero3 = hasattr(gpc.config, 'zero')

    # pipelinable = PipelinableContext()
    # with pipelinable:
    #     model = gpc.config.model.pop('type')(**gpc.config.model)

    # def mask_function(attention_mask=None):
    #     if attention_mask is not None:
    #         batch_size = gpc.config.BATCH_SIZE // gpc.config.NUM_MICRO_BATCHES
    #         attention_mask = attention_mask.view(batch_size, -1)
    #         attention_mask = col_nn.partition_batch(attention_mask)
    #         attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    #         attention_mask = (1.0 - attention_mask) * -10000.0
    #     return attention_mask

    #     # GPT2_small exec_seq
    #     # (lyl)TODO: The exec_seq for gpt3 will be added here and to_layer_list should be more friendly to use.
    # exec_seq = ['embed', mask_function, 'blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', (mask_function, "front"), \
    #             'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10', 'blocks.11', 'norm', 'head']
    # pipelinable.to_layer_list(exec_seq)
    # model = pipelinable.partition(num_chunks, gpc.pipeline_parallel_size,
    #                                 gpc.get_local_rank(ParallelMode.PIPELINE))


    # numel = calc_local_model_size(model)

    if not use_pipeline:
        ctx = nullcontext()
        if use_zero3:
            print("!!!!!!!!!!!!!!!!!!!!zero3")
            ctx = ZeroInitContext(target_device=torch.cuda.current_device(),
                                  shard_strategy=gpc.config.zero.model_config.shard_strategy,
                                  shard_param=True)
        with ctx:
            model = gpc.config.model.pop('type')(**gpc.config.model)
    else:
        pipelinable = PipelinableContext()
        with pipelinable:
            model = gpc.config.model.pop('type')(**gpc.config.model)

        def mask_function(attention_mask=None):
            # import pdb
            # pdb.set_trace()
            if attention_mask is not None:
                batch_size = gpc.config.BATCH_SIZE // gpc.config.NUM_MICRO_BATCHES
                attention_mask = attention_mask.view(batch_size, -1)
                attention_mask = col_nn.partition_batch(attention_mask)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = (1.0 - attention_mask) * -10000.0
            return attention_mask

        # GPT2_small exec_seq
        # (lyl)TODO: The exec_seq for gpt3 will be added here and to_layer_list should be more friendly to use.
        # exec_seq = ['embed', mask_function, 'blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', (mask_function, "front"), \
        #             'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10', 'blocks.11', 'norm', 'head']
        # exec_seq = ['embed', 'blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', \
        #             'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10', 'norm', 'head']          # gpt2-small
        # exec_seq = ['embed', 'blocks.0', \
        #             'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', \
        #             'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10', \
        #             'blocks.11', 'blocks.12', 'blocks.13', 'blocks.14', 'blocks.15', \
        #             'blocks.16', 'blocks.17', 'blocks.18', 'blocks.19', 'blocks.20', \
        #             'blocks.21', 'blocks.22', 'blocks.23', 'blocks.24', 'blocks.25', \
        #             'blocks.26', 'blocks.27', 'blocks.28', 'blocks.29', 'blocks.30', \
        #             'blocks.31', 'blocks.32', 'blocks.33', 'blocks.34', 'blocks.35', \
        #             'blocks.36', 'blocks.37', 'blocks.38', 'blocks.39', 'blocks.40', \
        #             'blocks.41', 'blocks.42', 'blocks.43', 'blocks.44', 'blocks.45', \
        #             'blocks.46','norm', 'head'] # gpt2_xl
        exec_seq = ['embed', 'blocks.0', \
                    'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', \
                    'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10', \
                    'blocks.11', 'blocks.12', 'blocks.13', 'blocks.14', 'blocks.15', \
                    'blocks.16', 'blocks.17', 'blocks.18', 'blocks.19', 'blocks.20', \
                    'blocks.21', 'blocks.22', 'blocks.23', 'blocks.24', 'blocks.25', \
                    'blocks.26', 'blocks.27', 'blocks.28', 'blocks.29', 'blocks.30', \
                    'blocks.31', 'blocks.32', 'blocks.33', 'blocks.34', 'blocks.35', \
                    'blocks.36', 'blocks.37', 'blocks.38', 'blocks.39', 'blocks.40', \
                    'blocks.41', 'blocks.42', 'blocks.43', 'blocks.44', 'blocks.45', \
                    'blocks.46', 'blocks.47', 'blocks.48', 'blocks.49', 'blocks.50', \
                    'blocks.51', 'blocks.52', 'blocks.53', 'blocks.54', 'blocks.55', \
                    'blocks.56', 'blocks.57', 'blocks.58', 'blocks.59', 'blocks.60', \
                    'blocks.61', 'blocks.62', 'norm', 'head'] # gpt2_4B
        # exec_seq = ['embed', 'blocks.0', \
        #             'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', \
        #             'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10', \
        #             'blocks.11', 'blocks.12', 'blocks.13', 'blocks.14', 'blocks.15', \
        #             'blocks.16', 'blocks.17', 'blocks.18', 'blocks.19', 'blocks.20', \
        #             'blocks.21', 'blocks.22', 'blocks.23', 'blocks.24', 'blocks.25', \
        #             'blocks.26', 'blocks.27', 'blocks.28', 'norm', 'head'] # gpt2_6B跑不起来
        # exec_seq = ['embed', 'blocks.0', \
        #             'blocks.1', 'blocks.2', 'blocks.3', 'blocks.4', 'blocks.5', \
        #             'blocks.6', 'blocks.7', 'blocks.8', 'blocks.9', 'blocks.10', \
        #             'blocks.11', 'blocks.12', 'blocks.13', 'blocks.14', 'blocks.15', \
        #             'blocks.16', 'blocks.17', 'blocks.18', 'blocks.19', 'blocks.20', \
        #             'blocks.21', 'blocks.22', 'blocks.23', 'blocks.24', 'blocks.25', \
        #             'blocks.26', 'blocks.27', 'blocks.28', 'blocks.29', 'blocks.30', \
        #             'blocks.31', 'blocks.32', 'blocks.33', 'blocks.34', 'blocks.35', \
        #             'blocks.36', 'blocks.37', 'blocks.38', 'blocks.39', 'blocks.40', \
        #             'blocks.41', 'blocks.42', 'blocks.43', 'blocks.44', 'blocks.45', \
        #             'blocks.46', 'blocks.47', 'blocks.48', 'blocks.49', 'blocks.50', \
        #             'blocks.51', 'blocks.52', 'blocks.53', 'blocks.54', 'blocks.55', \
        #             'blocks.56', 'blocks.57', 'blocks.58', 'blocks.59', 'blocks.60', \
        #             'blocks.61', 'blocks.62', 'blocks.63', 'blocks.64', 'blocks.65', \
        #             'blocks.66', 'blocks.67', 'blocks.68', 'blocks.69', 'blocks.70', \
        #             'norm', 'head'] # gpt2_8B
        pipelinable.to_layer_list(exec_seq)
        ctx = nullcontext()
        # (lyl)TODO: Zero context and pipelinable context should be integrated into one context.
        if use_zero3:
            ctx = ZeroInitContext(target_device=torch.cuda.current_device(),
                                  shard_strategy=gpc.config.zero.model_config.shard_strategy,
                                  shard_param=True)
        with ctx:
            model = pipelinable.partition(num_chunks, gpc.pipeline_parallel_size,
                                          gpc.get_local_rank(ParallelMode.PIPELINE))

    if use_zero3:
        numel = ctx.model_numel_tensor.item()
    else:
        numel = calc_local_model_size(model)

    tflop = numel * gpc.config.BATCH_SIZE * gpc.config.SEQ_LEN * 8 / (1024**4)
    criterion = getattr(gpc.config, 'loss_fn', None)
    if criterion is not None:
        criterion = criterion.type()
    else:
        criterion = GPTLMLoss()

    # logger.info('Build optimizer', ranks=[0])
    # optimizer = gpc.config.optimizer.pop('type')(model.parameters(), **gpc.config.optimizer)

    # lr_scheduler = LinearWarmupLR(optimizer, total_steps=gpc.config.NUM_EPOCHS, warmup_steps=5)

    engine, _, test_dataloader, _ = colossalai.initialize(model,
                                                          optimizer=None,
                                                          criterion=criterion,
                                                          test_dataloader=test_dataloader,
                                                          lr_scheduler=None)
    global_batch_size = gpc.config.BATCH_SIZE * \
        gpc.get_world_size(ParallelMode.DATA) * getattr(gpc.config, "gradient_accumulation", 1)
    logger.info(f'Init done, global batch size = {global_batch_size}', ranks=[0])

    timier = MultiTimer()

    trainer = Trainer(engine=engine, logger=logger, timer=timier)

    hook_list = [
        hooks.LossHook(),
        # hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=True),
        hooks.LogMetricByEpochHook(logger),
        hooks.ThroughputHook(ignored_steps=10, tflop_per_step=tflop),
        hooks.LogMetricByStepHook(),
    # hooks.TensorboardHook(log_dir='./tb_logs', ranks=[0]),
        hooks.LogMemoryByEpochHook(logger),
    # hooks.LogTimingByEpochHook(timer, logger),
    # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
    ]

    trainer.evaluate(test_dataloader=test_dataloader,
                hooks=hook_list,
                display_progress=True,
                return_output_label=False)


if __name__ == '__main__':
    main()