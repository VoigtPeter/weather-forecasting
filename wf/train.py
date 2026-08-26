import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from utils.config import Config
from wf.scaffold import ForecastModule


DEBUG = False


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    config = Config.from_yaml("../configs/train.yml")
    if DEBUG:
        config.dataset.train_time_slice = dict(start="1996", stop="1996")

    #module = ForecastModule.load_from_checkpoint("./checkpoints/iter_0.ckpt", config=config)
    config.dataset.train_seq_len = 3
    module = ForecastModule(config)
    module.train_forecast_steps = 1  # 1. train using single forecast step


    csv_logger = CSVLogger("./")
    checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints_01", monitor="val/loss_step1")
    trainer = L.Trainer(max_steps=config.trainer.max_steps, accelerator="auto", logger=csv_logger,
                        enable_checkpointing=True, callbacks=[checkpoint_callback],
                        val_check_interval=1.0, limit_val_batches=20)
    trainer.fit(module)
    print({k: round(float(v), 4) for k, v in trainer.callback_metrics.items()})
    torch.save(module.state_dict(), "./test_model_2p8_1_step.pt")


    # 2. restart training with 2 steps
    module = ForecastModule.load_from_checkpoint(checkpoint_callback.best_model_path, config=config)
    module.train_forecast_steps = 2

    csv_logger = CSVLogger("./")
    checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints_02", monitor="val/loss_step1")
    trainer = L.Trainer(max_steps=config.trainer.max_steps, accelerator="auto", logger=csv_logger,
                        enable_checkpointing=True, callbacks=[checkpoint_callback],
                        val_check_interval=1.0, limit_val_batches=20)
    trainer.fit(module)
    print({k: round(float(v), 4) for k, v in trainer.callback_metrics.items()})
    torch.save(module.state_dict(), "./test_model_2p8_2_step_finetuned.pt")
