import os.path
from typing import Literal
from dataclasses import dataclass, field, asdict

import json
import yaml

from wf.data.dataset import WeatherDatasetConfig
from wf.models.vit import ViTConfig

_t_model_cfg = ViTConfig
_MODEL_TYPES = {
    "vit": ViTConfig,
}


class Config:
    def __init__(self) -> None:
        self.dataset: WeatherDatasetConfig = None
        self.model: _t_model_cfg = None
        self.trainer: TrainerConfig = None
        self.loss: LossConfig = None

    @classmethod
    def from_dict(cls, cfg: dict, cfg_path: str) -> "Config":
        res = Config()
        top_level_keys = (("dataset", WeatherDatasetConfig), ("model", None), ("trainer", TrainerConfig), ("loss", LossConfig))
        for key, cfg_cls in top_level_keys:
            if key not in cfg or (cfg_cls is None and key != "model"):
                continue
            sub_cfg = cfg.get(key)
            if isinstance(sub_cfg, str):
                if not os.path.isabs(sub_cfg):
                    # relative path to main cfg file
                    sub_cfg = os.path.join(os.path.dirname(cfg_path), sub_cfg)
                assert os.path.isfile(sub_cfg)
                with open(sub_cfg) as f:
                    sub_cfg = yaml.safe_load(f)
            assert isinstance(sub_cfg, dict)

            if key == "model":
                cfg_cls = _MODEL_TYPES[sub_cfg.pop("type").lower()]
            res.__setattr__(key, cfg_cls(**sub_cfg))
        return res

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            cfg: dict = yaml.safe_load(f)
            assert isinstance(cfg, dict)
            return Config.from_dict(cfg, path)

    def to_dict(self) -> dict:
        res = dict()
        for key in ("dataset", "model", "trainer", "loss"):
            sub_cfg = getattr(self, key)
            if sub_cfg is not None:
                res[key] = asdict(sub_cfg)
        return res

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class TrainerConfig:
    lr: float
    batch_size: int = 1
    ensemble_size: int = 1
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    max_steps: int = 300
    num_data_workers: int = 0


@dataclass
class LossConfig:
    name: str
    kwargs: dict | None = None


if __name__ == "__main__":
    print(Config.from_yaml(r"C:\Users\voigt\PycharmProjects\weather-forecasting\configs\train.yml"))