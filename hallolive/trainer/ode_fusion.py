import gc
import logging
from hallolive.utils.dataset import ODEFusionLMDBDataset, cycle
from hallolive.model import ODEFusionRegression
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

        assert config.distribution_loss == "ode_fusion", "Only ODE loss is supported for ODE training"
        self.model = ODEFusionRegression(config, device=self.device)

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

        self.trainable_params = [param for param in self.model.generator.parameters() if param.requires_grad]

        self.generator_optimizer = torch.optim.AdamW(
            self.trainable_params, lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset = ODEFusionLMDBDataset(config.data_path, max_pair=getattr(config, "max_pair", int(1e8)))
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        self.dataloader = cycle(dataloader)

        self.step = 0

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        # if getattr(config, "generator_ckpt", False):
        #     print(f"Loading pretrained generator from {config.generator_ckpt}")
        #     state_dict = torch.load(config.generator_ckpt, map_location="cpu")["generator"]
        #     self.model.generator.load_state_dict(state_dict, strict=True)

        ##############################################################################################################

        self.max_grad_norm = 10.0
        self.previous_time = None

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

    def train_one_step(self):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        # Step 1: Get the next batch of text prompts
        batch = next(self.dataloader)
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

        loss_breakdown = defaultdict(list)
        stats = {}

        for index, t in enumerate(timestep):
            loss_breakdown[str(int(t.item()) // 250 * 250)].append(unnormalized_loss[index].item())

        for key_t in loss_breakdown.keys():
            stats["loss_at_time_" + key_t] = sum(loss_breakdown[key_t]) / len(loss_breakdown[key_t])

        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        if is_fsdp_wrapped(self.model.generator):
            generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
        else:
            generator_grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
        self.generator_optimizer.step()

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

            if (not self.config.no_save) and self.step % self.config.save_ckpt_steps == 0:
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
