import argparse
from omegaconf import OmegaConf
import wandb
from hallolive.trainer.dual_stream_ode_trainer import Trainer as DualStreamODETrainer
from hallolive.trainer.dual_stream_dmd_trainer import Trainer as DualStreamDMDTrainer
from hallolive.utils.config import add_args_to_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--save_ckpt_dir", type=str, default=None)

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config = add_args_to_config(config, args)

    if config.trainer == "dual_stream_ode":
        trainer = DualStreamODETrainer(config)
    elif config.trainer == "dual_stream_dmd":
        trainer = DualStreamDMDTrainer(config)
    else:
        raise ValueError(f"Unknown trainer: {config.trainer}")
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
