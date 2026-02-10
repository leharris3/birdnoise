# **FINCH**: Adaptive Evidence Weighting for Audio-Spatiotemporal Fusion

[![arXiv](https://img.shields.io/badge/arXiv-2602.03817-b31b1b.svg)](https://arxiv.org/abs/2602.03817)

Official implementation of **FINCH**, a framework for bioacoustic species identification that fuses audio classification with spatiotemporal priors from [eBird](https://ebird.org/) abundance data. We pair a frozen [NatureLM-audio](https://github.com/david-tedjopurnomo/NatureLM-audio) encoder with learned gating networks that adaptively weight audio evidence against space-time context.

## Installation

You'll need Python 3.10+, a [HuggingFace](https://huggingface.co/) account with access to [Meta Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) (required by NatureLM-audio), and a CUDA GPU (recommended).

```bash
cd NatureLM-audio
pip install -r requirements.txt
pip install scikit-learn seaborn h5py rasterio matplotlib pandas soundfile resampy tqdm wandb

# GPU support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# HuggingFace login
huggingface-cli login
```

You also need the BirdCLEF training data from [Kaggle](https://www.kaggle.com/c/birdclef-2021/data), organized as:

```
Data/
 cbi/
  train.csv          # must have: species, ebird_code, filename, latitude, longitude, date
  train_audio/
    abythr1/
      XC123456.mp3
    amerob/
      ...
```

The precomputed eBird prior cache (`Data/cbi/priors_cache_temp_1766584745.h5`) is already included -- no need to download eBird TIF files.

## Training

Training is split into two stages, both via `Scripts/train_weighted_fusion.py`.

**Stage A** learns a single scalar weight `w` for fusing audio logits with the spatiotemporal prior (the NatureLM encoder stays frozen):

```bash
cd Scripts

python train_weighted_fusion.py \
    --stage A \
    --data_dir ../Data/cbi \
    --priors_cache ../Data/cbi/priors_cache_temp_1766584745.h5 \
    --save_dir ../checkpoints_fusion_stage_a \
    --epochs 30 --batch_size 64 --lr 0.001 \
    --weight_decay 0.01 --warmup_ratio 0.1 --val_split 0.1 \
    --pooling mean --device cuda --num_workers 4
```

**Stage B** replaces the scalar weight with a gating network `w(a, x, t)` that adapts the fusion weight based on audio confidence, prior confidence, location, and time:

```bash
python train_weighted_fusion.py \
    --stage B \
    --data_dir ../Data/cbi \
    --priors_cache ../Data/cbi/priors_cache_temp_1766584745.h5 \
    --save_dir ../checkpoints_fusion \
    --epochs 30 --batch_size 96 --lr 0.001 \
    --weight_decay 0.01 --warmup_ratio 0.1 --val_split 0.1 \
    --pooling mean --w_max 2.0 --gate_hidden_dim 64 \
    --device cuda --num_workers 4
```

To warm-start Stage B from a Stage A checkpoint, add `--resume ../checkpoints_fusion_stage_a/checkpoint_epoch30.pth`. Checkpoints are saved every 5 epochs; the best model (by val loss) is saved as `best_model.pth`.

## Evaluation

`Scripts/evaluate_models.py` benchmarks four configurations -- prior-only, audio-only, Stage A posterior, and Stage B posterior -- reporting Probe accuracy, R-AUC, and NMI.

```bash
python evaluate_models.py \
    --data_dir ../Data/cbi \
    --priors_cache ../Data/cbi/priors_cache_temp_1766584745.h5 \
    --checkpoint_stage_a ../checkpoints_fusion_stage_a/checkpoint_epoch30.pth \
    --checkpoint_stage_b ../checkpoints_fusion/best_model.pth \
    --output_dir ../evaluation_results \
    --batch_size 64 --device cuda --num_workers 4
```

Results are saved to `evaluation_results/`: a metrics table (`comparison_table.png`), confusion matrices, pathological examples, and a CSV with all numbers.

## Notes

- **Prior cache**: the included `.h5` file has precomputed priors for every training sample. You never need to run `precompute_priors.py` or download eBird TIF rasters.
- **Data split**: 90/10 train/val by default (`--val_split 0.1`). Evaluation runs on the val set.
- **Reproducibility**: seed defaults to 42.
- **Logging**: training logs to [W&B](https://wandb.ai/) by default; pass `--no_wandb` to disable.
- **OOM**: lower `--batch_size` (try 32 or 16) if you hit memory limits.

## Citation

```bibtex
@article{ovanger2026adaptive,
    title   = {Adaptive Evidence Weighting for Audio-Spatiotemporal Fusion},
    author  = {Oscar Ovanger and Levi Harris and Timothy H. Keitt},
    journal = {arXiv preprint arXiv:2602.03817},
    year    = {2026},
    url     = {https://arxiv.org/abs/2602.03817}
}
```

## License

Released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
