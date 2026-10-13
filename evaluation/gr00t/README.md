# gr00t

ManiSkill clients for fine-tuned GR00T N1.7 policies. The model runs in its own environment as an Isaac-GR00T zmq server; `groot_client.py` reimplements the wire format (msgpack + `np.save` arrays), so the client environment does not need the `gr00t` package.

Observation sent to the server: `video.image` (base camera), `video.wrist_image` (hand camera), `state.arm` (7 joint positions), `state.gripper` (1), and the instruction under `annotation.human.task_description`. The returned `arm` + `gripper` chunk is executed as `pd_joint_pos` actions.

| File | Purpose |
|---|---|
| `start_server.sh` | `gr00t.eval.run_gr00t_server` for one checkpoint (`--embodiment-tag new_embodiment`). Needs `GR00T_REPO`. |
| `groot_conflict_eval.py` | Conflict (FDR) rollouts; driven by `../common/run_conflict.sh gr00t ...`. |
| `groot_grid_eval.py` + `run_pairwise_grid.sh` | Pairwise-grid eval (HARD protocol) for color_size, color_spatial, verb_spatial, spatial_size, spatial_object, verb_size. `COLOR_SIZE_MODE=easy` gives the easier color_size variant (1 distractor, ~2x size contrast). |
| `groot_full_factor_main.py` | All-Factor eval for a single 5-factor cell. |
| `groot_full_factor_batch.py` + `run_all_factor.sh` | All-Factor eval over 200 sampled cells in one process (same per-cell logic as `groot_full_factor_main.py`). |

```bash
GR00T_REPO=/path/to/Isaac-GR00T CUDA_VISIBLE_DEVICES=0 bash gr00t/start_server.sh /ckpts/color_spatial_stair_f18 5555 &

PORT=5555 bash ../common/run_conflict.sh gr00t color_spatial 42 200 results/conflict_gr00t
PORT=5555 bash gr00t/run_pairwise_grid.sh color_spatial results/grid/color_spatial_stair_f18 "42 40 41"
PORT=5555 bash gr00t/run_all_factor.sh results/all_factor/<ckpt> "40 41 42"
```

Checkpoints: FDR / FDH `yqi19/behaivor-gr00t-ckpt` (one folder per factor pair), pairwise grid `yqi19/gr00t_public_pair_ckpt`, All-Factor `yqi19/gr00t_all_factor_evaluation`.
