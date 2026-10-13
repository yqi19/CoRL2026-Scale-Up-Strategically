# vlm_judge

| File | Purpose |
|---|---|
| `vlm_eval.py` | Sends each conflict rollout video (`video/ep000.mp4`) and a factor-specific prompt to Gemini-2.5-Flash and records which factor the policy followed. Resumable JSONL output. |
| `fdr_fdh.py` | FDR per factor pair and the FDH Copeland ranking from the judge output (or from env success counts). |

The prompt depends on the pair type:
- verb pairs (verb_color, verb_object, verb_size, verb_spatial): did the robot perform the instructed verb, or the verb trained with the competing attribute?
- two-object pairs (the other six): the scene has object A (matches factor 1) and object B (matches factor 2); which one did the robot go for?

The answer field `FACTOR_FOLLOWED` is stored as `vlm_factor_followed` (factor 1 name, factor 2 name, or `neither`; `error` if the call failed).

```bash
export GEMINI_API_KEY=...          # or GOOGLE_API_KEY
python vlm_judge/vlm_eval.py --results-root results/conflict_gr00t --label gr00t \
    [--experiments verb_color color_size] [--workers 10] [--limit N]
python vlm_judge/fdr_fdh.py --vlm-dir results/conflict_gr00t/vlm_eval/gr00t --tau 0.05
python vlm_judge/fdr_fdh.py --env-dir results/conflict_gr00t      # env-success FDR
```

The default backend is the native Gemini REST API with the video sent inline. `--backend openai` posts the same request to an OpenAI-compatible chat endpoint (`VLM_API_URL`, default Gemini's OpenAI-compatible URL). `VLM_MODEL` overrides the model name (default `gemini-2.5-flash`).

FDR = (N_f1 - N_f2) / (N_f1 + N_f2) over rollouts labelled with f1 or f2. FDH: for each of the 10 pairs, the factor with FDR margin above tau = 0.05 wins, below -tau loses, otherwise tie; factors are ranked by wins minus losses. Labels are matched exactly; `--normalize-labels` also accepts bracketed or quoted labels such as `["color"]`.
