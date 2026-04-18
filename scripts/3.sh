#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/3.sh
#
#uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch "model.encoder_dims=[31, 128, 128, 64]" "model.decoder_dims=[128, 128, 128]" logger=aim logger.aim.experiment=DE
#
#uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch "model.encoder_dims=[31, 256, 256, 64]" "model.decoder_dims=[128, 256, 256]" logger=aim logger.aim.experiment=DE

#uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch "model.encoder_dims=[31, 512, 512, 64]" "model.decoder_dims=[128, 512, 512]" logger=aim logger.aim.experiment=DE




uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch "model.encoder_dims=[31, 512, 512, 16]" "model.decoder_dims=[128, 512, 512]" logger=aim logger.aim.experiment=DE1

uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch "model.encoder_dims=[31, 512, 512, 32]" "model.decoder_dims=[128, 512, 512]" logger=aim logger.aim.experiment=DE1

#uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch "model.encoder_dims=[31, 512, 512, 64]" "model.decoder_dims=[128, 512, 512]" logger=aim logger.aim.experiment=DE

uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch "model.encoder_dims=[31, 512, 512, 128]" "model.decoder_dims=[128, 512, 512]" logger=aim logger.aim.experiment=DE1







