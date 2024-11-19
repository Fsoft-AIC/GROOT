decoder_type = "cnn"

"""
======================
===== VAE CONFIG =====
======================
"""
# Encoder
encoder_kwargs = {
    "pretrained_encoder_path": "facebook/esm2_t30_150M_UR50D",
    "num_unfreeze_layers": 1
}
# Latent
latent_kwargs = {
    "latent_dim": 320
}
# Decoder
decoder_kwargs = {
    "dec_hidden_dim": 512,
}
# Predictor
predictor_kwargs = {
    "pred_hidden_dim": 512,
    "pred_dropout": 0.2,
}
# Others
model_kwargs = {
    "max_len": None,
    "nll_weight": 1.0,
    "mse_weight": 1.0,
    "kl_weight": 1.0,
    "beta_min": 0.0,
    "beta_max": 1.0,
    "Kp": 0.01,   # 0.01
    "Ki": 0.0001,
    "lr": 0.0001,
    "reduction": "mean",
    "interp_size": 16,
    "interp_weight": 0.005,
    "use_interp_sampling": False,
    "neg_focus": False,
    "neg_size": 16,
    "neg_norm": 2.0,
    "neg_weight": 0.1,
    "use_neg_sampling": False,
    "regularize_latent": False,
    "latent_weight": 0.001,
}

freeze_encoder = True


"""
======================
===== DATAMODULE =====
======================
"""
data_kwargs = {
    "train_val_split": (0.95, 0.05),
    "num_workers": 64,
    "standardize": False,
}


"""
=================
===== OTHER =====
=================
"""
wandb_project = "Smoothing"
num_ckpts = 3
save_every_n_epochs = 1
precision = "highest"
