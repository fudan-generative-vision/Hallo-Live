import gc
import logging
from contextlib import nullcontext
from hallolive.utils.dataset import DualStreamODELMDBDataset, cycle
from hallolive.model import DualStreamODEModel
from collections import defaultdict
from hallolive.utils.misc import set_seed, count_params
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import shutil
import wandb
import time
import os
from tqdm import tqdm


from hallolive.utils.distributed import barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job, is_fsdp_wrapped


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode=config.wandb_mode,
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir,
            )

        self.output_path = os.path.join(config.save_ckpt_dir, config.exp_name)

        if self.is_main_process:
            os.makedirs(self.output_path, exist_ok=True)
            shutil.copy(config.config_path, self.output_path)

        # Step 2: Initialize the model and optimizer

        assert config.distribution_loss == "dual_stream_ode", "Only ODE loss is supported for ODE training"
        self.model = DualStreamODEModel(config, device=self.device)

        if self.is_main_process:
            print(f"Generator parameters: {count_params(self.model.generator) / 1e9:.1f}B")
            print(f"Text ecnoder parameters: {count_params(self.model.text_encoder) / 1e9:.1f}B")

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
        )

        # Print FSDP wrapping structure
        # if self.is_main_process:
        #     print(self.model.generator)

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False),
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32
            )

        def build_stream_param_groups(module, base_lr, video_lr, audio_lr):
            stream_param_groups = {
                "video": {"params": [], "lr": video_lr},
                "audio": {"params": [], "lr": audio_lr},
                "shared": {"params": [], "lr": base_lr},
            }
            stream_param_counts = {"video": 0, "audio": 0, "shared": 0}

            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue

                if "video_model." in name:
                    stream_name = "video"
                elif "audio_model." in name:
                    stream_name = "audio"
                else:
                    stream_name = "shared"

                stream_param_groups[stream_name]["params"].append(param)
                stream_param_counts[stream_name] += param.numel()

            optimizer_param_groups = []
            for stream_name in ("video", "audio", "shared"):
                if stream_param_groups[stream_name]["params"]:
                    optimizer_param_groups.append(stream_param_groups[stream_name])

            return optimizer_param_groups, stream_param_counts

        generator_video_lr = getattr(config, "lr_video", config.lr)
        generator_audio_lr = getattr(config, "lr_audio", config.lr)
        generator_param_groups, generator_param_counts = build_stream_param_groups(
            self.model.generator, base_lr=config.lr, video_lr=generator_video_lr, audio_lr=generator_audio_lr
        )

        if self.is_main_process:
            print("--- Optimizer Stream LR ---")
            print(
                f"Generator video lr: {generator_video_lr:.3e} ({generator_param_counts['video'] / 1e6:.4f} M params)"
            )
            print(
                f"Generator audio lr: {generator_audio_lr:.3e} ({generator_param_counts['audio'] / 1e6:.4f} M params)"
            )
            print(f"Generator shared lr: {config.lr:.3e} ({generator_param_counts['shared'] / 1e6:.4f} M params)")

        self.trainable_params = [param for group in generator_param_groups for param in group["params"]]

        self.generator_optimizer = torch.optim.AdamW(
            generator_param_groups, lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset = DualStreamODELMDBDataset(config.data_path, max_pair=getattr(config, "max_pair", int(1e8)))
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)

        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        global_micro_batch_size = config.batch_size * self.world_size
        self.total_batch_size = global_micro_batch_size * self.gradient_accumulation_steps

        if self.is_main_process:
            print(f"ODE dataset size: {len(dataset):,}")
            print(
                f"Gradient accumulation steps: {self.gradient_accumulation_steps} "
                f"(global micro batch size: {global_micro_batch_size}, total batch size: {self.total_batch_size})"
            )

        self.dataloader = cycle(dataloader)

        self.step = 0
        self.max_grad_norm = 10.0
        self.previous_time = None

    def _gradient_sync_context(self, module, sync_gradients):
        if sync_gradients or self.gradient_accumulation_steps == 1 or not hasattr(module, "no_sync"):
            return nullcontext()
        return module.no_sync()

    def save(self):
        # print("Start gathering distributed model states...")
        if is_fsdp_wrapped(self.model.generator):
            generator_state_dict = fsdp_state_dict(self.model.generator)
        else:
            generator_state_dict = self.model.generator.state_dict()
        state_dict = {"generator": generator_state_dict}

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}", "model.pt"))

    def _fwdbwd_one_micro_step(self, batch):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]

        video_ode_latent = batch["ode_video_latent"].to(device=self.device, dtype=self.dtype)
        audio_ode_latent = batch["ode_audio_latent"].to(device=self.device, dtype=self.dtype)

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)

            conditional_dict["video_prompt_embeds"] = conditional_dict["prompt_embeds"]
            conditional_dict["audio_prompt_embeds"] = conditional_dict["prompt_embeds"]
            conditional_dict.pop("prompt_embeds")

        # Step 3: Train the generator
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
            generator_loss, log_dict = self.model.generator_loss(
                video_ode_latent=video_ode_latent, audio_ode_latent=audio_ode_latent, conditional_dict=conditional_dict
            )

        unnormalized_loss = log_dict["unnormalized_loss"]
        timestep = log_dict["timestep"]

        if self.world_size > 1:
            gathered_unnormalized_loss = torch.zeros(
                [self.world_size, *unnormalized_loss.shape], dtype=unnormalized_loss.dtype, device=self.device
            )
            gathered_timestep = torch.zeros(
                [self.world_size, *timestep.shape], dtype=timestep.dtype, device=self.device
            )

            dist.all_gather_into_tensor(gathered_unnormalized_loss, unnormalized_loss)
            dist.all_gather_into_tensor(gathered_timestep, timestep)
        else:
            gathered_unnormalized_loss = unnormalized_loss
            gathered_timestep = timestep

        (generator_loss / self.gradient_accumulation_steps).backward()

        return {
            "generator_loss": generator_loss.detach(),
            "unnormalized_loss": unnormalized_loss,
            "timestep": timestep,
        }

    def train_one_step(self):
        self.generator_optimizer.zero_grad(set_to_none=True)

        micro_step_outputs = []
        for micro_step in range(self.gradient_accumulation_steps):
            batch = next(self.dataloader)
            sync_gradients = micro_step == self.gradient_accumulation_steps - 1
            with self._gradient_sync_context(self.model.generator, sync_gradients):
                micro_step_outputs.append(self._fwdbwd_one_micro_step(batch))

        if is_fsdp_wrapped(self.model.generator):
            generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
        else:
            generator_grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
        self.generator_optimizer.step()

        generator_loss_values = []
        unnormalized_losses = []
        timesteps = []
        for output in micro_step_outputs:
            generator_loss_values.append(output["generator_loss"])
            unnormalized_losses.append(output["unnormalized_loss"])
            timesteps.append(output["timestep"])

        generator_loss = torch.stack(generator_loss_values).mean()
        unnormalized_loss = torch.cat(unnormalized_losses, dim=0)
        timestep = torch.cat(timesteps, dim=0)

        loss_breakdown = defaultdict(list)
        for index, t in enumerate(timestep):
            loss_breakdown[str(int(t.item()) // 250 * 250)].append(unnormalized_loss[index].item())
        stats = {f"loss_at_time_{key_t}": sum(values) / len(values) for key_t, values in loss_breakdown.items()}

        # Step 4: Logging
        if self.is_main_process and not self.disable_wandb:
            wandb_loss_dict = {
                "generator_loss": generator_loss.item(),
                "generator_grad_norm": generator_grad_norm.item(),
                **stats,
            }
            wandb.log(wandb_loss_dict, step=self.step)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

    def train(self):
        progress_bar = tqdm(
            initial=self.step, total=getattr(self.config, "max_steps", None), disable=not self.is_main_process
        )

        while True:
            self.train_one_step()

            if (not self.config.no_save) and self.step > 0 and self.step % self.config.save_ckpt_steps == 0:
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.step += 1

            progress_bar.update(1)

            if self.is_main_process:
                progress_bar.set_description("ODE initialization")

            if hasattr(self.config, "max_steps") and self.step >= (self.config.max_steps + 1):
                progress_bar.close()
                break

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
