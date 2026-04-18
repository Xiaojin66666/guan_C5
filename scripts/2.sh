#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/2.sh

uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch model.shared_layers=1 model.task_specific_layers=2 logger=aim logger.aim.experiment=shared_layer

uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch model.shared_layers=2 model.task_specific_layers=1 logger=aim logger.aim.experiment=shared_layer

#uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch model.shared_layers=1 model.task_specific_layers=3 logger=aim logger.aim.experiment=shared_layer

uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch model.shared_layers=2 model.task_specific_layers=2 logger=aim logger.aim.experiment=shared_layer

uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch model.shared_layers=3 model.task_specific_layers=1 logger=aim logger.aim.experiment=shared_layer

