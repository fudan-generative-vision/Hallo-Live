"""
Run a simulated multi-GPU workload that mimics model training.

Usage:
    python tools/fake_train.py
"""

import math
import os
import random
import subprocess
import time
import torch
import torch.multiprocessing as mp


def clamp(value, low, high):
    return max(low, min(high, value))


def check_mem(cuda_device):
    # Prefer nvidia-smi so the numbers match what operators usually watch.
    # Fall back to the CUDA API only if the shell command is unavailable.
    command = ["/usr/bin/nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,nounits,noheader"]
    try:
        output = subprocess.check_output(command, text=True).strip().splitlines()
        total, used = output[int(cuda_device)].split(",")
        return int(total.strip()), int(used.strip())
    except Exception:
        with torch.cuda.device(cuda_device):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        total_mb = total_bytes // 1024**2
        used_mb = total_mb - free_bytes // 1024**2
        return total_mb, used_mb


def pick_workspace_shape(total_mb):
    # Scale the reusable compute workspace to the card size.
    # Larger cards can afford wider matrices and more concurrent batches,
    # which makes the synthetic workload look closer to a real training job.
    if total_mb >= 64 * 1024:
        return 8, 2048
    if total_mb >= 32 * 1024:
        return 8, 1792
    if total_mb >= 20 * 1024:
        return 6, 1536
    if total_mb >= 12 * 1024:
        return 4, 1280
    return 3, 1024


def build_workspace(device, total_mb):
    # Preallocate reusable tensors so the runtime footprint stays stable.
    # This avoids constant allocator churn and keeps the "active training"
    # portion separate from the "reserved memory" portion.
    max_batch, max_dim = pick_workspace_shape(total_mb)
    dtype = torch.float16

    activations = torch.randn((max_batch, max_dim, max_dim), device=device, dtype=dtype)
    weights = torch.randn((max_batch, max_dim, max_dim), device=device, dtype=dtype)
    residual = torch.randn((max_batch, max_dim, max_dim), device=device, dtype=dtype)
    scratch = torch.empty_like(activations)
    grads = torch.empty_like(activations)
    params = torch.randn((max_dim, max_dim), device=device, dtype=dtype)
    momentum = torch.randn_like(params)
    updates = torch.empty_like(params)

    return {
        "device": device,
        "dtype": dtype,
        "max_batch": max_batch,
        "max_dim": max_dim,
        "activations": activations,
        "weights": weights,
        "residual": residual,
        "scratch": scratch,
        "grads": grads,
        "params": params,
        "momentum": momentum,
        "updates": updates,
    }


def reserve_device_memory(device, memory_fraction):
    # Hold most VRAM with a safety margin, then generate utilization separately.
    # Reserving memory in chunks is more tolerant of fragmentation than trying
    # to allocate one giant tensor in a single call.
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    total_mb = total_bytes // 1024**2
    safety_margin_mb = min(768, max(256, int(total_mb * 0.05)))
    target_used_bytes = int(total_bytes * memory_fraction)
    current_used_bytes = total_bytes - free_bytes
    reserve_bytes = max(0, target_used_bytes - current_used_bytes - safety_margin_mb * 1024**2)

    reserve_dtype = torch.float32
    bytes_per_elem = torch.tensor([], dtype=reserve_dtype).element_size()
    chunk_bytes = 256 * 1024**2
    reserved = []
    allocated_bytes = 0

    while reserve_bytes > 0:
        requested_bytes = min(reserve_bytes, chunk_bytes)
        requested_numel = requested_bytes // bytes_per_elem
        if requested_numel <= 0:
            break

        try:
            reserved.append(torch.empty(requested_numel, dtype=reserve_dtype, device=device))
            actual_bytes = requested_numel * bytes_per_elem
            allocated_bytes += actual_bytes
            reserve_bytes -= actual_bytes
        except RuntimeError:
            # If allocation fails, back off gradually instead of giving up
            # immediately. This makes the script more robust on busy machines.
            if chunk_bytes <= 32 * 1024**2:
                break
            chunk_bytes = max(32 * 1024**2, chunk_bytes // 2)
            torch.cuda.empty_cache()

    return reserved, allocated_bytes // 1024**2


class TrainingRhythm:
    def __init__(self, rng):
        self.rng = rng
        self.next_eval = rng.randint(140, 220)
        self.eval_steps_remaining = 0
        self.next_checkpoint = rng.randint(90, 150)

    def sample_step(self, step):
        # Alternate between warmup, steady training, eval dips, and checkpoint stalls.
        # The goal is not perfect realism, but to avoid a flat utilization trace
        # that obviously looks like a fixed synthetic benchmark.
        cycle_pos = step % 160
        if cycle_pos < 12:
            # Simulate early-step ramp-up when kernels, caches, and data flow
            # have not reached their long-running steady state yet.
            base_load = 0.30 + cycle_pos * 0.04
        else:
            # Stay around a high steady-state load with two overlapping waves
            # so the curve has short and medium-period variation.
            base_load = 0.76 + 0.11 * math.sin(step / 6.0) + 0.05 * math.sin(step / 17.0)

        checkpoint_pause = 0.0
        if step >= self.next_checkpoint:
            # Periodic checkpointing creates a visible pause that operators are
            # used to seeing in real training loops.
            checkpoint_pause = self.rng.uniform(0.25, 0.9)
            self.next_checkpoint += self.rng.randint(90, 150)

        if self.eval_steps_remaining > 0:
            # Evaluation usually runs at lower throughput than training because
            # it changes the batch pattern and may include host-side work.
            base_load *= self.rng.uniform(0.45, 0.65)
            self.eval_steps_remaining -= 1
        elif step >= self.next_eval:
            # Enter a short multi-step eval window instead of a single dip so
            # the graph looks more natural.
            self.eval_steps_remaining = self.rng.randint(2, 5)
            self.next_eval += self.rng.randint(150, 240)
            base_load *= self.rng.uniform(0.55, 0.7)

        # Add noise so repeated runs on the same machine do not follow the
        # exact same utilization trace.
        base_load += self.rng.uniform(-0.08, 0.08)
        base_load = clamp(base_load, 0.28, 0.97)

        # Higher load usually correlates with more gradient accumulation or
        # more work packed into a single training step.
        micro_steps = self.rng.randint(1, 3 if base_load < 0.65 else 5)
        data_wait = self.rng.uniform(0.01, 0.05)
        if self.rng.random() < 0.18:
            # Occasional input stalls imitate dataloader jitter or CPU-side
            # preprocessing hiccups.
            data_wait += self.rng.uniform(0.04, 0.2)

        # Short communication waits mimic distributed synchronization without
        # actually requiring multiple ranks or NCCL traffic.
        comm_wait = self.rng.uniform(0.003, 0.018) if self.rng.random() < 0.75 else 0.0
        forward_budget = self.rng.uniform(0.04, 0.09) * micro_steps * (0.75 + base_load * 0.55)
        backward_budget = forward_budget * self.rng.uniform(0.95, 1.35)
        optimizer_budget = self.rng.uniform(0.01, 0.03)

        return {
            "target_load": base_load,
            "micro_steps": micro_steps,
            "data_wait": data_wait,
            "comm_wait": comm_wait,
            "forward_budget": forward_budget,
            "backward_budget": backward_budget,
            "optimizer_budget": optimizer_budget,
            "checkpoint_pause": checkpoint_pause,
        }


def pick_dim(max_dim, intensity, rng):
    # Pick matrix sizes from a small ladder rather than a continuous range.
    # That creates repeated "model-like" kernels instead of totally random shapes.
    dim_choices = [384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048]
    dim_choices = [dim for dim in dim_choices if dim <= max_dim]
    anchor = int(round((len(dim_choices) - 1) * clamp(intensity, 0.0, 1.0)))
    lower = max(0, anchor - 1)
    upper = min(len(dim_choices) - 1, anchor + 1)
    return dim_choices[rng.randint(lower, upper)]


def pick_batch(max_batch, intensity, stage, rng):
    if stage == "optimizer":
        # Optimizer-style work is represented as parameter updates, not as a
        # large batched activation tensor.
        return 1

    scaled = clamp(intensity + rng.uniform(-0.08, 0.08), 0.15, 1.0)
    estimated = max(1, int(round(scaled * max_batch)))
    # Backward gets slightly more jitter to make it look heavier and less uniform
    # than forward on utilization graphs.
    jitter = rng.randint(0, 1 if stage == "forward" else 2)
    return min(max_batch, max(1, estimated + jitter))


def run_forward_block(state, batch, dim, mix_ratio):
    # Approximate a forward pass with batched GEMMs and a simple activation path.
    x = state["activations"][:batch, :dim, :dim]
    y = state["weights"][:batch, :dim, :dim]
    residual = state["residual"][:batch, :dim, :dim]
    out = state["scratch"][:batch, :dim, :dim]

    torch.bmm(x, y, out=out)
    out.add_(residual, alpha=mix_ratio)
    out.relu_()


def run_backward_block(state, batch, dim, grad_scale):
    # Reuse the forward result and run another GEMM to imitate backward-time
    # gradient propagation and post-processing.
    out = state["scratch"][:batch, :dim, :dim]
    residual = state["residual"][:batch, :dim, :dim]
    grad = state["grads"][:batch, :dim, :dim]

    torch.bmm(out, residual, out=grad)
    grad.mul_(grad_scale)
    grad.tanh_()


def run_optimizer_block(state, dim):
    # Keep optimizer work cheaper than backward, but still GPU-visible.
    params = state["params"][:dim, :dim]
    momentum = state["momentum"][:dim, :dim]
    updates = state["updates"][:dim, :dim]

    torch.mm(params, momentum, out=updates)
    params.add_(updates, alpha=1e-3)


def run_compute_burst(state, duration_s, intensity, stage, rng):
    if duration_s <= 0:
        return

    # Different stages use different effective intensity so the curve is less uniform.
    # Forward, backward, and optimizer should not look equally expensive.
    stage_scale = {"forward": 0.9, "backward": 1.0, "optimizer": 0.55}[stage]
    intensity = clamp(intensity * stage_scale, 0.2, 1.0)
    deadline = time.perf_counter() + duration_s

    while True:
        # Re-sample shape each burst so the workload looks like changing batch
        # composition or sequence length across steps.
        dim = pick_dim(state["max_dim"], intensity, rng)
        batch = pick_batch(state["max_batch"], intensity, stage, rng)

        if stage == "forward":
            run_forward_block(state, batch, dim, rng.uniform(0.03, 0.12))
        elif stage == "backward":
            run_forward_block(state, batch, dim, rng.uniform(0.02, 0.08))
            run_backward_block(state, batch, dim, rng.uniform(0.82, 1.08))
        else:
            run_optimizer_block(state, dim)

        # Synchronize intentionally so each burst consumes roughly the requested
        # wall-clock duration instead of queueing far ahead asynchronously.
        torch.cuda.synchronize(state["device"])
        if time.perf_counter() >= deadline:
            break


def loop(cuda_device):
    # Each process owns one GPU and continuously emits a training-like pattern.
    torch.cuda.set_device(cuda_device)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(f"cuda:{cuda_device}")
    total_mb, _ = check_mem(cuda_device)
    workspace = build_workspace(device, total_mb)
    # Keep a reference so the allocator holds the reserved VRAM.
    reserved_tensors, reserved_mb = reserve_device_memory(device, memory_fraction=0.9)

    total_mb, used_mb = check_mem(cuda_device)
    print(
        f"Device number: {cuda_device} | Total memory: {total_mb} MB | "
        f"Reserved by this script: {reserved_mb} MB | Current used: {used_mb} MB"
    )

    seed = int(time.time()) + os.getpid() + cuda_device * 1009
    rng = random.Random(seed)
    rhythm = TrainingRhythm(rng)
    step = 0

    while True:
        # Build one fake training iteration with realistic waits and compute phases.
        profile = rhythm.sample_step(step)

        # Host-side delay before compute starts. This is where a real training
        # loop would often be waiting for input data.
        time.sleep(profile["data_wait"])

        per_micro_forward = profile["forward_budget"] / profile["micro_steps"]
        for _ in range(profile["micro_steps"]):
            # Multiple forward micro-steps imitate gradient accumulation.
            run_compute_burst(
                workspace,
                duration_s=per_micro_forward,
                intensity=profile["target_load"] * rng.uniform(0.9, 1.05),
                stage="forward",
                rng=rng,
            )

        if profile["comm_wait"] > 0:
            # Brief gaps here make the trace resemble all-reduce or pipeline
            # coordination waits.
            time.sleep(profile["comm_wait"])

        run_compute_burst(
            workspace,
            duration_s=profile["backward_budget"],
            intensity=profile["target_load"] * rng.uniform(1.0, 1.1),
            stage="backward",
            rng=rng,
        )
        run_compute_burst(
            workspace,
            duration_s=profile["optimizer_budget"],
            intensity=profile["target_load"],
            stage="optimizer",
            rng=rng,
        )

        if profile["checkpoint_pause"] > 0:
            # Simulate blocking work such as saving checkpoints to disk.
            time.sleep(profile["checkpoint_pause"])

        step += 1


def main():
    if torch.cuda.is_available():
        # Mirror the original behavior and occupy every visible GPU.
        num_processes = torch.cuda.device_count()
        processes = []
        for i in range(num_processes):
            process = mp.Process(target=loop, args=(i,))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
