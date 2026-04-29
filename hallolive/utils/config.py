import os


def add_args_to_config(config, args):
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    if "config_path" in config and config.config_path is not None:
        config.config_name = os.path.basename(config.config_path).split(".")[0]

    return config
