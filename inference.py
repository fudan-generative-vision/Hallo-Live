import argparse
import os
import time
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler, Subset

import hallolive.utils.memory as memory_utils
from hallolive.pipeline import causal_inference_fusion
from hallolive.pipeline import CausalInferenceFusionPipeline
from hallolive.utils.config import add_args_to_config
from hallolive.utils.dataset import TextDataset, TextImagePairDataset
from hallolive.utils.misc import set_seed
from hallolive.ovi.utils.io_utils import save_video
from hallolive.ovi.utils.processing_utils import format_prompt_for_filename
from hallolive.utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb


def setup_distributed(seed: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference.")

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Some imported modules cache the target GPU during import time, so patch them after setting the rank device.
    memory_utils.gpu = device
    causal_inference_fusion.gpu = device

    set_seed(seed)
    return device, local_rank, global_rank, world_size


def sample_noise(sample_seeds, shape, device, dtype):
    noise = []
    for sample_seed in sample_seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(sample_seed)
        noise.append(torch.randn(shape, generator=generator, device=device, dtype=dtype))
    return torch.stack(noise, dim=0)


def main(config):
    device, local_rank, global_rank, world_size = setup_distributed(config.seed)

    print(f"[Rank {global_rank}] Free VRAM {get_cuda_free_memory_gb(device):.2f} GB")
    low_memory = get_cuda_free_memory_gb(device) < 40

    torch.set_grad_enabled(False)

    # Initialize pipeline
    t0 = time.time()
    generator_ckpt = getattr(config, "generator_ckpt", None)
    config.generator_meta_init = bool(generator_ckpt)
    pipeline = CausalInferenceFusionPipeline(config, device=device)
    print(f"[Timer] Pipeline initialization: {time.time() - t0:.2f} seconds")

    if generator_ckpt:
        t0 = time.time()
        state_dict = torch.load(generator_ckpt, map_location="cpu", mmap=True, weights_only=True)
        print(f"[Timer] Load checkpoint from disk: {time.time() - t0:.2f} seconds")

        if config.use_ema and "generator_ema" in state_dict:
            state_dict = state_dict["generator_ema"]
            print("Load generator EMA version")
        elif "generator" in state_dict:
            state_dict = state_dict["generator"]
            print("Load generator original version")
        state_dict = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in state_dict.items()}

        t1 = time.time()
        pipeline.generator.load_state_dict(
            state_dict, assign=config.generator_meta_init
        )  # model.video_model.patch_embedding.weight
        print(f"[Timer] Load state_dict into model: {time.time() - t1:.2f} seconds")

    t0 = time.time()
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device, dtype=torch.bfloat16)
    if config.generator_meta_init:
        pipeline.generator.model.set_rope_params()
    pipeline.vae_video.to(device=device, dtype=torch.bfloat16)
    pipeline.vae_audio.to(device=device, dtype=torch.bfloat16)
    print(f"[Timer] Move models to GPU: {time.time() - t0:.2f} seconds")

    # Create dataset
    if config.i2v:
        transform = transforms.Compose(
            [transforms.Resize((480, 832)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        dataset = TextImagePairDataset(config.data_path, transform=transform)
    else:
        dataset = TextDataset(prompt_path=config.data_path)
    num_prompts = len(dataset)
    if world_size == 1:
        rank_dataset = dataset
    else:
        rank_indices = list(range(global_rank, len(dataset), world_size))
        rank_dataset = Subset(dataset, rank_indices)
    print(f"[Rank {global_rank}] Number of prompts: {num_prompts}, assigned to this rank: {len(rank_dataset)}")

    sampler = SequentialSampler(rank_dataset)
    dataloader = DataLoader(rank_dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

    # Create output directory (only on main process to avoid race conditions)
    if local_rank == 0:
        os.makedirs(config.output_folder, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    for i, batch_data in tqdm(enumerate(dataloader), total=len(dataloader), disable=(local_rank != 0)):
        idx = batch_data["idx"].item()
        sample_seeds = [config.seed + offset for offset in range(config.num_samples)]

        # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
        # Unpack the batch data for convenience
        if isinstance(batch_data, dict):
            batch = batch_data
        elif isinstance(batch_data, list):
            batch = batch_data[0]  # First (and only) item in the batch

        if config.i2v:
            # For image-to-video, batch contains image and caption
            prompt = batch["prompts"][0]  # Get caption from batch
            prompts = [prompt] * config.num_samples

            # Process the image
            image = batch["image"].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

            # Encode the input image as the first latent
            t_vae_encode = time.time()
            initial_latent = pipeline.vae_video.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(config.num_samples, 1, 1, 1, 1)
            print(f"[Timer] Sample {i} - VAE encode initial image: {time.time() - t_vae_encode:.2f} seconds")

            video_noise = sample_noise(
                sample_seeds, (config.num_output_frames - 1, 48, 32, 62), device=device, dtype=torch.bfloat16
            )
            audio_noise = sample_noise(sample_seeds, (150, 20), device=device, dtype=torch.bfloat16)
        else:
            # For text-to-video, batch is just the text prompt
            prompt = batch["prompts"][0]
            prompts = [prompt] * config.num_samples
            initial_latent = None

            video_noise = sample_noise(
                sample_seeds, (config.num_output_frames, 48, 32, 62), device=device, dtype=torch.bfloat16
            )

            audio_noise = sample_noise(sample_seeds, (150, 20), device=device, dtype=torch.bfloat16)

        # Pipeline inference timing
        t_inference = time.time()
        video, audio, video_latents, audio_latents = pipeline.inference(
            video_noise=video_noise,
            audio_noise=audio_noise,
            text_prompts=prompts,
            return_latents=True,
            profile=config.profile,
            initial_latent=initial_latent,
            low_memory=low_memory,
        )
        torch.cuda.synchronize(device)  # Ensure all CUDA operations are complete
        print(f"[Timer] Sample {i} - Pipeline inference: {time.time() - t_inference:.2f} seconds")

        # Post-processing timing
        video = video.cpu().float().numpy()
        audio = audio.cpu().float().numpy()

        # Clear VAE cache
        pipeline.vae_video.model.clear_cache()

        video_frame_height_width = [512, 992]

        # Save video timing
        # Save the video if the current prompt is not a dummy prompt
        if idx < num_prompts:
            for seed_idx, sample_seed in enumerate(sample_seeds):
                # All processes save their videos
                formatted_prompt = format_prompt_for_filename(prompt)
                output_path = os.path.join(
                    config.output_folder,
                    f"{formatted_prompt}_{'x'.join(map(str, video_frame_height_width))}_{sample_seed}.mp4",
                )
                save_video(output_path, video[seed_idx], audio[seed_idx].squeeze(), fps=24, sample_rate=16000)

        print("-" * 60)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to the model directory")
    parser.add_argument("--generator_ckpt", type=str, help="Path to the generator checkpoint")
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--output_folder", type=str, help="Output folder")
    parser.add_argument(
        "--num_output_frames", type=int, default=30, help="Number of overlap frames between sliding windows"
    )
    parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
    parser.add_argument("--profile", action="store_true", help="Whether to profile")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config = add_args_to_config(config, args)
    config.seed = int(config.seed)
    config.num_output_frames = int(config.num_output_frames)
    config.num_samples = int(config.num_samples)

    main(config)
