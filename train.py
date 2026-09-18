import os

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from wf.utils.config import Config
from wf.scaffold import ForecastModule


DEBUG = False


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    model_dir = "2p8_WeT_sfno"
    config = Config.from_yaml("./configs/wet_train_2p8_sfno.yml")
    val_check_interval = 0.33

    if DEBUG:
        model_dir = "debug"
        config.trainer.max_steps = 100
        val_check_interval = 20
        config.dataset.train_time_slice = dict(start="2006", stop="2006")

    #module = ForecastModule.load_from_checkpoint("./checkpoints/iter_0.ckpt", config=config)
    """config.dataset.train_seq_len = 2
    module = ForecastModule(config)
    module.train_forecast_steps = 1  # 1. train using single forecast step

    # compile
    #print("compiling model...")
    module.model.compile(fullgraph=False, mode="max-autotune")
    
    csv_logger = CSVLogger(f"logs/{model_dir}")
    checkpoint_callback = ModelCheckpoint(dirpath=f"./logs/{model_dir}/checkpoints/step_1", monitor="val/loss_step3")
    trainer = L.Trainer(max_steps=config.trainer.max_steps, accelerator="auto", logger=csv_logger,
                        enable_checkpointing=True, callbacks=[checkpoint_callback],
                        val_check_interval=val_check_interval, limit_val_batches=20)
    trainer.fit(module)
    print({k: round(float(v), 4) for k, v in trainer.callback_metrics.items()})#"""

    # --------------------------------
    # 2. restart training with N steps
    # --------------------------------
    ROLLOUT_STEPS = 4
    BATCH_REDUCTION = 4
    LR_REDUCTION = 4
    val_check_interval = 100

    print("--- Multi-step forecast finetuning ---")
    config.trainer.batch_size = config.trainer.batch_size // BATCH_REDUCTION
    config.trainer.lr = (config.trainer.lr / LR_REDUCTION) / BATCH_REDUCTION
    config.dataset.train_seq_len = ROLLOUT_STEPS + 1
    config.trainer.max_steps = config.trainer.max_steps // ROLLOUT_STEPS

    ckpt = torch.load("./logs/2p8_WeT_sfno/checkpoints/step_1/epoch=14-step=17439.ckpt")
    #ckpt = torch.load(checkpoint_callback.best_model_path)
    modified_state_dict = dict()
    for key, value in ckpt["state_dict"].items():
        key: str = key.replace("._orig_mod", "")
        modified_state_dict[key] = value
    module: ForecastModule = ForecastModule(config)
    module.load_state_dict(modified_state_dict)
    module.train_forecast_steps = ROLLOUT_STEPS

    # compile
    module.model.compile(fullgraph=False, mode="max-autotune")

    csv_logger = CSVLogger(f"./logs/{model_dir}")
    checkpoint_callback = ModelCheckpoint(dirpath=f"./logs/{model_dir}/checkpoints/step_{ROLLOUT_STEPS}_ft", monitor="val/loss_step3")
    trainer = L.Trainer(max_steps=config.trainer.max_steps, accelerator="auto", logger=csv_logger,
                        enable_checkpointing=True, callbacks=[checkpoint_callback],
                        val_check_interval=val_check_interval, limit_val_batches=20)
    trainer.fit(module)
    print({k: round(float(v), 4) for k, v in trainer.callback_metrics.items()})
    #torch.save(module.state_dict(), "./test_model_2p8_2_step_finetuned.pt")"""

    # go to sleep
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
