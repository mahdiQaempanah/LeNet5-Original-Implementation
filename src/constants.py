from pydantic import BaseModel
import yaml
import argparse


class ModelConfig(BaseModel):
    A: float
    S: float

class PathsConfig(BaseModel):
    rbf_file_format: str
    dataset_dir: str

class TrainingConfig(BaseModel):
    lr: float
    batch_size: int
    num_epochs: int
    device: str
    shuffle: bool
    
class AppConfig(BaseModel):
    model: ModelConfig
    paths: PathsConfig
    training: TrainingConfig


class ConfigHandler:
    
    @staticmethod
    def load_config(path: str = "../config.yaml") -> AppConfig:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        return AppConfig(**raw)

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument("--lr", type=float)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--num_epochs", type=int)
        parser.add_argument("--device", type=str)
        args, unknown = parser.parse_known_args()
        return args

    @staticmethod
    def apply_overrides(config: AppConfig, args) -> AppConfig:
        if args.lr is not None:
            config.training.lr = args.lr

        if args.batch_size is not None:
            config.training.batch_size = args.batch_size

        if args.num_epochs is not None:
            config.training.num_epochs = args.num_epochs

        if args.device is not None:
            config.training.device = args.device

        return config


config = ConfigHandler.load_config()
args = ConfigHandler.parse_args()
config = ConfigHandler.apply_overrides(config, args)

