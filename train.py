import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from wf.utils.config import Config
from wf.scaffold import ForecastModule


DEBUG = False


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    config = Config.from_yaml("./configs/wet_train_2p8_sfno.yml")
    if DEBUG:
        config.trainer.max_steps = 100
        config.dataset.train_time_slice = dict(start="2006", stop="2006")

    #module = ForecastModule.load_from_checkpoint("./checkpoints/iter_0.ckpt", config=config)
    config.dataset.train_seq_len = 2
    module = ForecastModule(config)
    module.train_forecast_steps = 1  # 1. train using single forecast step

    # compile
    #print("compiling model...")
    #module.model = torch.compile(module.model, fullgraph=False, mode="max-autotune")
    module.model.compile(fullgraph=False, mode="max-autotune")
    
    csv_logger = CSVLogger("logs/2p8_WeT_sfno_gcn")
    checkpoint_callback = ModelCheckpoint(dirpath="./logs/2p8_WeT_sfno_gcn/checkpoints/step_1", monitor="val/loss_step1")
    trainer = L.Trainer(max_steps=config.trainer.max_steps, accelerator="auto", logger=csv_logger,
                        enable_checkpointing=True, callbacks=[checkpoint_callback],
                        val_check_interval=1.0, limit_val_batches=20)
    trainer.fit(module)
    print({k: round(float(v), 4) for k, v in trainer.callback_metrics.items()})#"""


    # 2. restart training with 2 steps
    #module = ForecastModule.load_from_checkpoint(checkpoint_callback.best_model_path, config=config)

    """ROLLOUT_STEPS = 4
    BATCH_REDUCTION = 4
    LR_REDUCTION = 3

    config.trainer.batch_size = config.trainer.batch_size // BATCH_REDUCTION
    config.trainer.lr = (config.trainer.lr / LR_REDUCTION) / BATCH_REDUCTION
    config.dataset.train_seq_len = ROLLOUT_STEPS + 1
    config.trainer.max_steps = config.trainer.max_steps * BATCH_REDUCTION

    ckpt = torch.load("./logs/1p5_sfno_WeT/checkpoints/step_1/epoch=39-step=18240.ckpt")
    modified_state_dict = dict()
    for key, value in ckpt["state_dict"].items():
        key: str = key.replace("._orig_mod", "")
        modified_state_dict[key] = value
    module: ForecastModule = ForecastModule(config)
    module.load_state_dict(modified_state_dict)

    #module = ForecastModule.load_from_checkpoint("../logs/2p8_extra_features/checkpoints/step_2/epoch=20-step=19194.ckpt", config=config)

    module.train_forecast_steps = ROLLOUT_STEPS

    #module.model = torch.compile(module.model, fullgraph=True, mode="max-autotune")

    csv_logger = CSVLogger("./logs/1p5_sfno_WeT")
    checkpoint_callback = ModelCheckpoint(dirpath="./logs/1p5_sfno_WeT/checkpoints/step_4_ft", monitor="val/loss_step3")
    trainer = L.Trainer(max_steps=config.trainer.max_steps, accelerator="auto", logger=csv_logger,
                        enable_checkpointing=True, callbacks=[checkpoint_callback],
                        val_check_interval=1.0, limit_val_batches=20)
    trainer.fit(module)
    print({k: round(float(v), 4) for k, v in trainer.callback_metrics.items()})
    #torch.save(module.state_dict(), "./test_model_2p8_2_step_finetuned.pt")"""
