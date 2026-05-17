import gc
import logging
from contextlib import nullcontext

from hallolive.utils.dataset import ShardingLMDBDataset, cycle
from hallolive.utils.dataset import TextDataset
from hallolive.utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from hallolive.utils.misc import set_seed, merge_dict_list, count_params
import torch.distributed as dist
from omegaconf import OmegaConf
from hallolive.model import DMDFusion
import torch
import shutil
from tqdm import tqdm
import wandb
import time
import os


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
        self.causal = config.causal
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
        if config.distribution_loss == "dmd_fusion":
            self.model = DMDFusion(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss: only dmd_fusion is supported")

        self.model: DMDFusion = self.model

        if self.is_main_process:
            print(f"Generator parameters: {count_params(self.model.generator) / 1e9:.1f}B")
            print(f"Real score parameters: {count_params(self.model.real_score) / 1e9:.1f}B")
            print(f"Fake score parameters: {count_params(self.model.fake_score) / 1e9:.1f}B")
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

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False),
        )

        partial_finetune = getattr(config, "partial_finetune", False)
        train_audio_stream_only = getattr(config, "train_audio_stream_only", False)

        if partial_finetune or train_audio_stream_only:
            trainable_params = 0
            frozen_params = 0

            if partial_finetune:
                self.model.generator.requires_grad_(False)

            for name, param in self.model.generator.named_parameters():
                if train_audio_stream_only and "video_model." in name:
                    param.requires_grad = False
                elif partial_finetune:
                    is_trainable = False
                    for trainable_module_name in config.trainable_modules:
                        if trainable_module_name in name:
                            is_trainable = True
                            param.requires_grad = True
                            break
                    if not is_trainable:
                        param.requires_grad = False

                if param.requires_grad:
                    trainable_params += param.numel()
                else:
                    frozen_params += param.numel()

            if partial_finetune:
                self.model.fake_score.requires_grad_(False)

            for name, param in self.model.fake_score.named_parameters():
                if train_audio_stream_only and "video_model." in name:
                    param.requires_grad = False
                    continue
                if partial_finetune:
                    for trainable_module_name in config.trainable_modules:
                        if trainable_module_name in name:
                            param.requires_grad = True
                            break

            total_params = trainable_params + frozen_params

            if self.is_main_process:
                print("--- Parameter Statistics ---")
                print(f"Trainable params: {trainable_params:,} ({trainable_params / 1e6:.4f} M)")
                print(f"Frozen params:    {frozen_params:,} ({frozen_params / 1e6:.4f} M)")
                print(f"Total params:     {total_params:,} ({total_params / 1e6:.4f} M)")
                print(f"Trainable ratio:  {100 * trainable_params / total_params:.4f}%")

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
        critic_base_lr = config.lr_critic if hasattr(config, "lr_critic") else config.lr
        critic_video_lr = getattr(config, "lr_critic_video", critic_base_lr)
        critic_audio_lr = getattr(config, "lr_critic_audio", critic_base_lr)

        generator_param_groups, generator_param_counts = build_stream_param_groups(
            self.model.generator, base_lr=config.lr, video_lr=generator_video_lr, audio_lr=generator_audio_lr
        )
        critic_param_groups, critic_param_counts = build_stream_param_groups(
            self.model.fake_score, base_lr=critic_base_lr, video_lr=critic_video_lr, audio_lr=critic_audio_lr
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
            print(f"Critic video lr: {critic_video_lr:.3e} ({critic_param_counts['video'] / 1e6:.4f} M params)")
            print(f"Critic audio lr: {critic_audio_lr:.3e} ({critic_param_counts['audio'] / 1e6:.4f} M params)")
            print(f"Critic shared lr: {critic_base_lr:.3e} ({critic_param_counts['shared'] / 1e6:.4f} M params)")

        self.generator_optimizer = torch.optim.AdamW(
            generator_param_groups, lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            critic_param_groups,
            lr=critic_base_lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay,
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)

        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        global_micro_batch_size = config.batch_size * self.world_size
        self.total_batch_size = global_micro_batch_size * self.gradient_accumulation_steps

        if self.is_main_process:
            print(f"Text prompts dataset size: {len(dataset):,}")
            print(
                f"Gradient accumulation steps: {self.gradient_accumulation_steps} "
                f"(global micro batch size: {global_micro_batch_size}, total batch size: {self.total_batch_size})"
            )
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        def rename_param(name):
            return (
                name.replace("_fsdp_wrapped_module.", "")
                .replace("_checkpoint_wrapped_module.", "")
                .replace("_orig_mod.", "")
            )

        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0) and (self.step >= config.ema_start_step):
            if self.is_main_process:
                print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def _gradient_sync_context(self, module, sync_gradients):
        if sync_gradients or self.gradient_accumulation_steps == 1 or not hasattr(module, "no_sync"):
            return nullcontext()
        return module.no_sync()

    def save(self):
        # print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(self.model.generator)
        critic_state_dict = fsdp_state_dict(self.model.fake_score)

        if self.config.ema_start_step < self.step and self.generator_ema is not None:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {"generator": generator_state_dict, "critic": critic_state_dict}

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]

        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1].to(device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        audio_shape = list(self.config.audio_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)

            conditional_dict["video_prompt_embeds"] = conditional_dict["prompt_embeds"]
            conditional_dict["audio_prompt_embeds"] = conditional_dict["prompt_embeds"]
            conditional_dict.pop("prompt_embeds")

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=(
                        [self.config.video_negative_prompt] * batch_size
                        + [self.config.audio_negative_prompt] * batch_size
                    )
                )
                unconditional_dict["video_prompt_embeds"] = unconditional_dict["prompt_embeds"][:batch_size]
                unconditional_dict["audio_prompt_embeds"] = unconditional_dict["prompt_embeds"][batch_size:]
                unconditional_dict.pop("prompt_embeds")

                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                audio_shape=audio_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                text_prompts=text_prompts,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
            )

            (generator_loss / self.gradient_accumulation_steps).backward()

            generator_log_dict.update({"generator_loss": generator_loss.detach()})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            audio_shape=audio_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None,
        )

        (critic_loss / self.gradient_accumulation_steps).backward()

        critic_log_dict.update({"critic_loss": critic_loss.detach()})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104], device="cuda", dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104], device="cuda", dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise, text_prompts=prompts, return_latents=True, initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        progress_bar = tqdm(
            initial=self.step, total=getattr(self.config, "max_steps", None), disable=not self.is_main_process
        )

        while True:
            if self.step % 20 == 0:
                torch.cuda.empty_cache()

            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                for micro_step in range(self.gradient_accumulation_steps):
                    batch = next(self.dataloader)
                    sync_gradients = micro_step == self.gradient_accumulation_steps - 1
                    with self._gradient_sync_context(self.model.generator, sync_gradients):
                        extra = self.fwdbwd_one_step(batch, True)
                    extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
                generator_log_dict["generator_grad_norm"] = generator_grad_norm.detach()
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            for micro_step in range(self.gradient_accumulation_steps):
                batch = next(self.dataloader)
                sync_gradients = micro_step == self.gradient_accumulation_steps - 1
                with self._gradient_sync_context(self.model.fake_score, sync_gradients):
                    extra = self.fwdbwd_one_step(batch, False)
                extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
            critic_log_dict["critic_grad_norm"] = critic_grad_norm.detach()
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            progress_bar.update(1)

            if self.is_main_process:
                progress_bar.set_description("DMD distillation")

            # Create EMA params (if not already created)
            if (
                (self.step >= self.config.ema_start_step)
                and (self.generator_ema is None)
                and (self.config.ema_weight > 0)
            ):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and self.step % self.config.save_ckpt_steps == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "video_dmdtrain_gradient_norm": generator_log_dict["video_dmdtrain_gradient_norm"]
                            .mean()
                            .item(),
                            "audio_dmdtrain_gradient_norm": generator_log_dict["audio_dmdtrain_gradient_norm"]
                            .mean()
                            .item(),
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item(),
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            if hasattr(self.config, "max_steps") and self.step >= self.config.max_steps:
                progress_bar.close()
                break

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
