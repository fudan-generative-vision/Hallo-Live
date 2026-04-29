from datetime import timedelta
from functools import partial
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import warnings
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    ModuleWrapPolicy,
    _or_policy,
)
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions


def print_fsdp_structure(model, indent=0):
    for name, child in model.named_children():
        is_wrapped = isinstance(child, FSDP)
        status = "[FSDP WRAPPED]" if is_wrapped else "[NOT WRAPPED]"
        print("  " * indent + f"{name}: {status}")
        # If it is not wrapped, keep descending to see whether any inner layers are wrapped
        if not is_wrapped:
            print_fsdp_structure(child, indent + 1)


def is_fsdp_wrapped(model):
    return isinstance(model, FSDP)


# def fsdp_state_dict(model, offload_to_cpu=True):
#     options = StateDictOptions(
#         full_state_dict=True,
#         cpu_offload=offload_to_cpu,
#     )
#     state_dict = get_state_dict(model, options=options)
#     print(state_dict.keys())
#     return state_dict


def fsdp_state_dict(model, offload_to_cpu=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)

        fsdp_fullstate_save_policy = FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy):
            checkpoint = model.state_dict()

    return checkpoint


def fsdp_wrap(
    module,
    sharding_strategy="full",
    mixed_precision=False,
    wrap_strategy="size",
    min_num_params=int(5e7),
    transformer_module=None,
    cpu_offload=False,
):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )
    else:
        mixed_precision_policy = None

    from hallolive.ovi.modules.causal_model import CausalWanSelfAttention, WanT2VCrossAttention, ModulationAdd, WanLayerNorm

    # from hallolive.ovi.modules.causal_fusion import CausalFusionModel
    from hallolive.ovi.modules.fusion import WanLayerNorm as OviWanLayerNorm
    from hallolive.ovi.modules.model import (
        ModulationAdd as OviModulationAdd,
        WanT2VCrossAttention as OviWanT2VCrossAttention,
        WanSelfAttention as OviWanSelfAttention,
    )

    transformer_module = [
        nn.Conv3d,
        nn.Sequential,
        CausalWanSelfAttention,
        WanT2VCrossAttention,
        ModulationAdd,
        WanLayerNorm,
    ]

    ovi_transformer_module = [OviWanT2VCrossAttention, OviWanSelfAttention, OviWanLayerNorm, OviModulationAdd]
    transformer_module += ovi_transformer_module

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_module)
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
    elif wrap_strategy == "cls":
        if isinstance(transformer_module, (list, tuple, set)):
            target_classes = set(transformer_module)
        else:
            target_classes = {transformer_module}
        auto_wrap_policy = ModuleWrapPolicy(target_classes)
    elif wrap_strategy == "hybrid":
        # Strategy A: Class-based wrapping (for specific custom layers)
        if isinstance(transformer_module, (list, tuple, set)):
            target_classes = tuple(transformer_module)
        else:
            target_classes = (transformer_module,)

        def lambda_policy(module, recurse, nonwrapped_numel):
            # Wrap if the module is an instance of the specified target classes
            return isinstance(module, target_classes)

        # Strategy B: Size-based wrapping (to prevent remaining layers from becoming too large)
        size_policy = partial(size_based_auto_wrap_policy, min_num_params=min_num_params)

        # Combined Strategy: Trigger wrapping if either Strategy A or Strategy B is satisfied
        auto_wrap_policy = partial(_or_policy, policies=[lambda_policy, size_policy])
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "full_zero2": ShardingStrategy.SHARD_GRAD_OP,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False,  # Load ckpt on rank 0 and sync to other ranks
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        init_method=init_method,
        timeout=timedelta(minutes=30),
    )
    torch.cuda.set_device(local_rank)


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.summon_full_params(fsdp_module, writeback=False):
            for n, p in fsdp_module.module.named_parameters():
                self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1.0 - d)

    # Optional helpers ---------------------------------------------------
    def state_dict(self):
        return self.shadow  # picklable

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, fsdp_module):
        # load EMA weights into an (unwrapped) copy of the generator
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.summon_full_params(fsdp_module, writeback=True):
            for n, p in fsdp_module.module.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n].to(p.dtype, device=p.device))
