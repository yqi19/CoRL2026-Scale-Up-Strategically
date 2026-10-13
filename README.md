# Scale Up Strategically

Code for **Scale Up Strategically: Learning Compositional Generalization via Bias-Aware Evaluation and Data Collection for Robotic Manipulation** (CoRL 2026).

[Project page](https://yqi19.github.io/scale-up-strategically/) · [Paper](https://arxiv.org/abs/2607.21582)

Fine-tuned robot policies over-rely on some instruction factors (e.g., color) and ignore others (e.g., verb, size). We measure this *instruction factor bias* with the Factor Dominance Rate (FDR) and Factor Dominance Hierarchy (FDH), and use it to decide where a fixed demonstration budget should go.

## Repository layout

| Folder | What it does |
|---|---|
| [`maniskill/`](maniskill) | ManiSkill 3 fork with our tabletop tasks (`VerbObjectColor-v1`) and motion-planning solvers. |
| [`data_collection/`](data_collection) | Demonstration collection: factor-correlated data for FDR, and the collection strategies (V, L, Random, Complete). Converts to LeRobot format. |
| [`finetune/`](finetune) | Fine-tuning recipes for GR00T-N1.7 and π0.5. |
| [`evaluation/`](evaluation) | Simulation evaluation and the VLM judge used to compute FDR / FDH. |

## Pipeline

```
maniskill (tasks)  ->  data_collection  ->  finetune  ->  evaluation
                       (demos, LeRobot)    (GR00T, π0.5)  (success rate, FDR / FDH)
```

Each folder has its own README with installation and commands.

## Data and checkpoints

Datasets and fine-tuned checkpoints are linked from the [project page](https://yqi19.github.io/scale-up-strategically/).

## Citation

```bibtex
@inproceedings{qi2026scaleup,
  title     = {Scale Up Strategically: Learning Compositional Generalization via
               Bias-Aware Evaluation and Data Collection for Robotic Manipulation},
  author    = {Qi, Yu and Ye, Zhang and Xu, Xinyi and Lu, Yuxuan and Sandhu, Amitoj and
               Hu, Boce and Huang, Haojie and Tremblay, Jonathan and Wong, Lawson L.S.},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```
