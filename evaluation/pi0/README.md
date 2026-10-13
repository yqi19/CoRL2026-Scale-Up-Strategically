# pi0

ManiSkill clients for pi0 / pi0.5 policies served by OpenPI (`scripts/serve_policy.py`, websocket). The client sends `observation/image` and `observation/wrist_image` (resized with padding to 224x224), `observation/state` (8-D joint state) and `prompt`, and executes the first `replan` actions of the returned chunk.

The OpenPI server side (the `pi0_maniskill` / `pi05_maniskill` configs and fine-tuning) is described in [`../../finetune`](../../finetune).

| File | Purpose |
|---|---|
| `start_server.sh` | `serve_policy.py` for one checkpoint. Needs `OPENPI_REPO`; third argument selects the config. |
| `pi0_conflict_eval.py` | Conflict (FDR) rollouts, same scenes and outputs as the GR00T version; driven by `../common/run_conflict.sh pi0 ...`. |
| `pi0_grid_eval.py` + `run_pairwise_grid.sh` | Pairwise-grid eval (HARD protocol) for color_size and color_spatial. Episodes end at the env time limit (`--max-episode-steps` is not applied in this client). |

```bash
OPENPI_REPO=/path/to/openpi CUDA_VISIBLE_DEVICES=0 bash pi0/start_server.sh /ckpts/color_spatial_stair_f18/17999 8000 pi0_maniskill &

PORT=8000 bash ../common/run_conflict.sh pi0 color_spatial 42 200 results/conflict_pi0
PORT=8000 bash pi0/run_pairwise_grid.sh color_spatial results/grid_pi0/color_spatial_stair_f18 "42 40 41"
```

ManiSkill runs on CPU by default here: running the pi0 server and ManiSkill GPU simulation on the same GPU caused CUDA illegal-access errors.
