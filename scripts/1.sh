#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/1.sh
uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch model.shared_layers=1

uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch model.shared_layers=2

uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch model.shared_layers=3

uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=stch model.shared_layers=4



uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=famo model.shared_layers=1

uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=famo model.shared_layers=2

uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=famo model.shared_layers=3

uv run src/train.py experiment=ftd model=ftd_mlp_mtl model.loss_term=famo model.shared_layers=4

