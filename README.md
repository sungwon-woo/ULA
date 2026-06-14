# ULA_FSCIL

Inference-time Adaptive Logit Adjustment built on top of the SAVC
backbone for Few-Shot Class-Incremental Learning (FSCIL). ULA shifts
probability mass from a base top-1 class to an incremental top-2 class when the
top-2 margin is small and the base-only entropy is high, without retraining the
backbone.

This repository extends the official implementation of the CVPR 2023 paper
*Learning with Fantasy: Semantic-Aware Virtual Contrastive Constraint for
Few-Shot Class-Incremental Learning* ([paper](https://arxiv.org/abs/2304.00426)).


## Requirements

- Python 3.8+
- [PyTorch >= 1.1 and torchvision](https://pytorch.org)
- tqdm
- numpy
- matplotlib (for confusion matrix saving)

## Datasets

CIFAR100, CUB200, and miniImageNet. Follow the data preparation guidelines in
[CEC](https://github.com/icoz69/CEC-CVPR2021).


## Pretrained Checkpoint

This repository performs **incremental-session evaluation only**: session 0
expects a pretrained SAVC checkpoint loaded via `-model_dir`. Train the base
session with the original SAVC repository or your own pipeline, then point
`-model_dir` to the resulting `session0_max_acc.pth`.

## Running Evaluation

The base commands match the original SAVC scripts; add `-model_dir <ckpt>` to
provide the session-0 weights.

- **miniImageNet**
  ```bash
  python train.py -project savc -dataset mini_imagenet \
    -base_mode 'ft_cos' -new_mode 'avg_cos' \
    -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 \
    -epochs_base 120 -schedule Milestone -milestones 40 70 100 \
    -gpu 0 -temperature 16 -moco_dim 128 -moco_k 8192 -mlp \
    -moco_t 0.07 -moco_m 0.999 \
    -size_crops 84 50 -min_scale_crops 0.2 0.05 -max_scale_crops 1.0 0.14 \
    -num_crops 2 4 -constrained_cropping \
    -alpha 0.2 -beta 0.8 -fantasy rot_color_perm12 \
    -model_dir <path/to/session0_max_acc.pth>
  ```

Use `-log_dir <dir>` to override the output directory and `-incft` for
incremental finetuning.


## Output

The trainer prints per-session metrics for both the unadjusted (`Orig`) and
adjusted (`ADJ`) logits. After incremental sessions, the last five lines of
stdout are, in order:

1. Average accuracy
2. Seen (base) accuracy
3. Unseen (incremental) accuracy
4. Generalized AUC
5. Harmonic mean of seen/unseen

## Acknowledgments

- [SAVC](https://github.com/zysong0113/SAVC) — backbone implementation.
- [CEC](https://github.com/icoz69/CEC-CVPR2021) — data splits.


## Citation

```bibtex
@inproceedings{woo2026training,
  title={Training-Free Uncertainty-guided Logit Adjustment for Few-Shot Class-Incremental Learning},
  author={Sungwon Woo and Dongjun Hwang and Shiwon Kim and Junsuk Choe and Jongho Nang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
