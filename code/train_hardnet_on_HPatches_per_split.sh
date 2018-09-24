#!/bin/bash

RUNPATH="$( cd "$(dirname "$0")" ; pwd -P )/.."
DATASETS="$RUNPATH/data/sets"
DATALOGS="$RUNPATH/data/logs"
DATAMODELS="$RUNPATH/data/models"

#mkdir -p "$DATASETS"
#mkdir -p "$DATALOGS"
#cd "$RUNPATH"
#echo "$DATASETS"
python3 -utt HardNetHPatchesSplits.py --hpatches-split="$DATASETS/hpatches_splits_full/hpatches_split_a_train.pt"  --gpu-id=0 --fliprot=True --model-dir="$DATAMODELS/model_HPatches_HardNet_a_lr01_trimar/" --dataroot "$DATASETS/PhotoTourism/" --lr=0.1 --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --n-triplets=15000000 --imageSize 32 --log-dir "$DATALOGS/log_HPatches_HardNet_a_lr01_trimar/"  --experiment-name=hpatches_a_aug_lr01/ | tee -a "$DATALOGS/hpatches/log_HardNet_orthlr01_aug_HPatches_a.log"
python3 -utt HardNetHPatchesSplits.py --hpatches-split="$DATASETS/hpatches_splits_full/hpatches_split_c_train.pt"  --gpu-id=0 --fliprot=True --model-dir="$DATAMODELS/model_HPatches_HardNet_c_lr01_trimar/" --dataroot "$DATASETS/PhotoTourism/" --lr=0.1 --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --n-triplets=15000000 --imageSize 32 --log-dir "$DATALOGS/log_HPatches_HardNet_c_lr01_trimar/"  --experiment-name=hpatches_c_aug_lr01/ | tee -a "$DATALOGS/hpatches/log_HardNet_orthlr01_aug_HPatches_c.log"
python3 -utt HardNetHPatchesSplits.py --hpatches-split="$DATASETS/hpatches_splits_full/hpatches_split_view_test.pt"  --gpu-id=0 --fliprot=True --model-dir="$DATAMODELS/model_HPatches_HardNet_view_lr01_trimar/" --dataroot "$DATASETS/PhotoTourism/" --lr=0.1 --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --n-triplets=15000000 --imageSize 32 --log-dir "$DATALOGS/log_HPatches_HardNet_view_lr01_trimar/"  --experiment-name=hpatches_view_aug_lr01/ | tee -a "$DATALOGS/hpatches/log_HardNet_orthlr01_aug_HPatches_view.log"
python3 -utt HardNetHPatchesSplits.py --hpatches-split="$DATASETS/hpatches_splits_full/hpatches_split_illum_test.pt"  --gpu-id=0 --fliprot=True --model-dir="$DATAMODELS/model_HPatches_HardNet_illum_lr01_trimar/" --dataroot "$DATASETS/PhotoTourism/" --lr=0.1 --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --n-triplets=15000000 --imageSize 32 --log-dir "$DATALOGS/log_HPatches_HardNet_illum_lr01_trimar/"  --experiment-name=hpatches_illum_aug_lr01/ | tee -a "$DATALOGS/hpatches/log_HardNet_orthlr01_aug_HPatches_illum.log"
python3 -utt HardNetHPatchesSplits.py --hpatches-split="$DATASETS/hpatches_splits_full/hpatches_split_b_train.pt"  --gpu-id=0 --fliprot=True --model-dir="$DATAMODELS/model_HPatches_HardNet_b_lr01_trimar/" --dataroot "$DATASETS/PhotoTourism/" --lr=0.1 --w1bsroot "$DATASETS/wxbs-descriptors-benchmark/code/" --n-triplets=15000000 --imageSize 32 --log-dir "$DATALOGS/log_HPatches_HardNet_b_lr01_trimar/"  --experiment-name=hpatches_b_aug_lr01/ | tee -a "$DATALOGS/hpatches/log_HardNet_orthlr01_aug_HPatches_b.log"
