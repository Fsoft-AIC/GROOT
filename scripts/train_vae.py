import argparse
import importlib
import os
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch import loggers, callbacks

from groot.common.utils import parse_module_name_from_path, parse_dict_from_module
from groot.models import CNNVAE
from groot.dataio.proteins import ProteinsDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE model.")
    parser.add_argument("config_file", type=str, help="Path to config module")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV data.")
    parser.add_argument("--expected_kl",
                        type=float,
                        default=40,
                        help="Expected KL-Divergence value.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--devices",
                        type=str,
                        default="-1",
                        help="Training devices separated by comma.")
    parser.add_argument("--epochs", type=int, default=150, help="# Training epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--ckpt_path", type=str, help="Checkpoint of model.")
    parser.add_argument("--wandb_id", type=str, help="WandB ID to resume.")
    parser.add_argument("--prefix", type=str, default="", help="Prefix of checkpoints.")
    parser.add_argument("--dataset", type=str, choices=["AAV", "GFP"], required=True)
    args = parser.parse_args()
    return args


def train(args):
    # Create cfg
    cfg = importlib.import_module(parse_module_name_from_path(args.config_file))
    # general config
    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.precision)
    accelerator = "cpu" if args.devices == "-1" else "gpu"
    device = torch.device("cuda" if accelerator == "gpu" else "cpu")

    # ================== #
    # ====== Data ====== #
    # ================== #
    data_kwargs = cfg.data_kwargs
    data_kwargs.update({
        "csv_data": args.csv_file,
        "seed": args.seed,
        "train_batch_size": args.batch_size,
        "valid_batch_size": args.batch_size,
    })
    datamodule = ProteinsDataModule(**data_kwargs)

    max_length = datamodule.max_length
    neg_floor = datamodule.min_fitness

    # =================== #
    # ====== Model ====== #
    # =================== #
    module_kwargs = {
        **cfg.encoder_kwargs,
        **cfg.latent_kwargs,
        **cfg.decoder_kwargs,
        **cfg.predictor_kwargs,
        **cfg.model_kwargs
    }
    module_kwargs.update({
        "expected_kl": args.expected_kl,
        "max_len": max_length,
        "device": device,
        "neg_floor": neg_floor,
    })
    module = CNNVAE(**module_kwargs)

    if cfg.freeze_encoder:
        module.freeze_encoder()     # freeze ESM2 encoder

    # ====================== #
    # ====== Training ====== #
    # ====================== #
    # Set up dirpath and naming rules
    data_name = args.dataset
    ckpt_filename = f"{cfg.decoder_type}-{args.prefix}-expkl={args.expected_kl}-vae-{data_name}_" \
        + "{epoch:02d}-{train_loss:.3f}-{valid_loss:.3f}"
    ckpt_dirpath = os.path.join(args.output_dir, f"vae_ckpts/{data_name}")
    os.makedirs(ckpt_dirpath, exist_ok=True)

    logger_list = [
        loggers.WandbLogger(
            save_dir=args.output_dir,
            id=args.wandb_id,
            project=cfg.wandb_project,
            config=parse_dict_from_module(cfg).update(args.__dict__),
            log_model=False,
            group=f"{cfg.decoder_type}-{data_name}"
        )
    ]
    callback_list = [
        callbacks.RichModelSummary(),
        callbacks.RichProgressBar(),
        callbacks.ModelCheckpoint(
            dirpath=ckpt_dirpath,
            filename=ckpt_filename,
            monitor="valid_loss",
            verbose=True,
            save_top_k=cfg.num_ckpts,
            save_weights_only=False,
            save_last=True,
            every_n_epochs=cfg.save_every_n_epochs,
        )
    ]

    trainer = Trainer(
        accelerator=accelerator,
        devices=[int(d) for d in args.devices.split(",")],
        max_epochs=args.epochs,
        default_root_dir=args.output_dir,
        logger=logger_list,
        callbacks=callback_list,
        strategy="ddp_find_unused_parameters_true" if len(args.devices.split(",")) > 1 else "auto",
        gradient_clip_val=None,
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    args = parse_args()
    train(args)
