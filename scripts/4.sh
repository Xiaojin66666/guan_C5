#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/1.sh

#uv run src/train.py experiment=ftd model=ftd_mlp_mtl 'data.altitude=["&H35000F0M69X21","&H35000F0M73X21","&H35000F0M80X21","&H35000F0M81X21","&H35000F0M785X21"]' logger=aim logger.aim.experiment=altitude
#
#
#uv run src/train.py experiment=ftd model=ftd_mlp_mtl data.action='LonStab' logger=aim logger.aim.experiment=action
uv run src/train.py experiment=ftd model=ftd_mlp_mtl data.action='BTB' model.optimizer.lr=0.0001 "model.encoder_dims=[31, 512, 512, 64]" model.task_specific_layers=1 logger=aim logger.aim.experiment=action

uv run src/train.py experiment=ftd model=ftd_mlp_mtl data.action='BTB' logger=aim logger.aim.experiment=action

uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term='' logger=aim logger.aim.experiment=nomtl


uv run src/train.py experiment=ftd model=ftd_mlp_mtl 'data.altitude=["&H35000F0M69X21","&H35000F0M73X21","&H35000F0M80X21","&H35000F0M81X21","&H35000F0M785X21"]' model.optimizer.lr=0.0001 logger=aim logger.aim.experiment=altitude

uv run src/train.py experiment=ftd model=ftd_mlp_mtl data.action='LonStab' model.optimizer.lr=0.0001 logger=aim logger.aim.experiment=action

uv run src/train.py experiment=ftd model=ftd_mlp_mtl data.action='BTB' model.optimizer.lr=0.0001 logger=aim logger.aim.experiment=action


uv run src/train.py experiment=ftd model=ftd_mlp_mtl 'data.altitude=["&H35000F0M69X21","&H35000F0M73X21","&H35000F0M80X21","&H35000F0M81X21","&H35000F0M785X21"]' model.optimizer.lr=0.0001 "model.encoder_dims=[31, 512, 512, 64]" logger=aim logger.aim.experiment=altitude

uv run src/train.py experiment=ftd model=ftd_mlp_mtl data.action='LonStab' model.optimizer.lr=0.0001 "model.encoder_dims=[31, 512, 512, 64]" logger=aim logger.aim.experiment=action

uv run src/train.py experiment=ftd model=ftd_mlp_mtl data.action='BTB' model.optimizer.lr=0.0001 "model.encoder_dims=[31, 512, 512, 64]" logger=aim logger.aim.experiment=action




uv run src/train.py experiment=ftd model=ftd_mlp_mtl 'data.altitude=["&H35000F0M69X21","&H35000F0M73X21","&H35000F0M80X21","&H35000F0M81X21","&H35000F0M785X21"]' model.optimizer.lr=0.0001 "model.encoder_dims=[31, 512, 512, 64]" model.task_specific_layers=1 logger=aim logger.aim.experiment=altitude

uv run src/train.py experiment=ftd model=ftd_mlp_mtl data.action='LonStab' model.optimizer.lr=0.0001 "model.encoder_dims=[31, 512, 512, 64]" model.task_specific_layers=1 logger=aim logger.aim.experiment=action

#uv run src/train.py experiment=ftd model=ftd_mlp_mtl data.action='BTB' model.optimizer.lr=0.0001 "model.encoder_dims=[31, 512, 512, 64]" model.task_specific_layers=1 logger=aim logger.aim.experiment=action





#
#uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch model.optimizer.lr=0.01 logger=aim logger.aim.experiment=lr
#
#uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch model.optimizer.lr=0.05 logger=aim logger.aim.experiment=lr
#
#uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch model.optimizer.lr=0.001 logger=aim logger.aim.experiment=lr
#
#uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch model.optimizer.lr=0.005 logger=aim logger.aim.experiment=lr
#
##uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch model.optimizer.lr=0.0001 logger=aim logger.aim.experiment=lr
#
#uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch model.optimizer.lr=0.00005 logger=aim logger.aim.experiment=lr
#
#uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch model.optimizer.lr=0.00001 logger=aim logger.aim.experiment=lr




#uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch "model.encoder_dims=[31, 512, 512, 128]" "model.decoder_dims=[128, 512, 512]" logger=aim logger.aim.experiment=mtl
#
#uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=famo "model.encoder_dims=[31, 512, 512, 128]" "model.decoder_dims=[128, 512, 512]" logger=aim logger.aim.experiment=mtl
#
#uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=fairgrad "model.encoder_dims=[31, 512, 512, 128]" "model.decoder_dims=[128, 512, 512]" logger=aim logger.aim.experiment=mtl
#
#
