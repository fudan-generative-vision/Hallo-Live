import torch
import os
import json
from pathlib import Path
from safetensors.torch import load_file

from hallolive.ovi.modules.fusion import FusionModel
from hallolive.ovi.modules.t5 import T5EncoderModel
from hallolive.ovi.modules.vae2_2 import Wan2_2_VAE
from hallolive.ovi.modules.mmaudio.features_utils import FeaturesUtils

OVI_ROOT = Path(__file__).resolve().parent.parent


def init_wan_vae_2_2(model_dir, rank=0):
    vae_config = {}
    vae_config["device"] = rank
    vae_pth = os.path.join(model_dir, "Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    vae_config["vae_pth"] = vae_pth
    vae_model = Wan2_2_VAE(**vae_config)

    return vae_model


def init_mmaudio_vae(model_dir, rank=0):
    vae_config = {}
    vae_config["mode"] = "16k"
    vae_config["need_vae_encoder"] = True

    tod_vae_ckpt = os.path.join(model_dir, "MMAudio/ext_weights/v1-16.pth")
    bigvgan_vocoder_ckpt = os.path.join(model_dir, "MMAudio/ext_weights/best_netG.pt")

    vae_config["tod_vae_ckpt"] = tod_vae_ckpt
    vae_config["bigvgan_vocoder_ckpt"] = bigvgan_vocoder_ckpt

    vae = FeaturesUtils(**vae_config).to(rank)

    return vae


def init_fusion_score_model_ovi(meta_init=False):
    video_config = OVI_ROOT / "configs" / "model" / "dit" / "video_5B.json"
    audio_config = OVI_ROOT / "configs" / "model" / "dit" / "audio_5B.json"
    assert video_config.exists(), f"{video_config} does not exist"
    assert audio_config.exists(), f"{audio_config} does not exist"

    with open(video_config) as f:
        video_config = json.load(f)

    with open(audio_config) as f:
        audio_config = json.load(f)

    if meta_init:
        with torch.device("meta"):
            fusion_model = FusionModel(video_config, audio_config)
    else:
        fusion_model = FusionModel(video_config, audio_config)

    return fusion_model, video_config, audio_config


def init_text_model(model_dir, rank, cpu_offload=False):
    wan_dir = os.path.join(model_dir, "Wan2.2-TI2V-5B")
    text_encoder_path = os.path.join(wan_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    text_tokenizer_path = os.path.join(wan_dir, "google/umt5-xxl")

    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=rank,
        checkpoint_path=text_encoder_path,
        tokenizer_path=text_tokenizer_path,
        cpu_offload=cpu_offload,
        shard_fn=None,
    )

    return text_encoder


def load_fusion_checkpoint(model, checkpoint_path, from_meta=False):
    if checkpoint_path and os.path.exists(checkpoint_path):
        if checkpoint_path.endswith(".safetensors"):
            df = load_file(checkpoint_path, device="cpu")
        elif checkpoint_path.endswith(".pt"):
            try:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                df = df["module"] if "module" in df else df
            except Exception as e:
                df = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                df = df["app"]["model"]
        else:
            raise RuntimeError("We only support .safetensors and .pt checkpoints")

        missing, unexpected = model.load_state_dict(df, strict=True, assign=from_meta)

        del df
        import gc

        gc.collect()
    else:
        raise RuntimeError("{checkpoint=} does not exists'")
