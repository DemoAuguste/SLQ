# Scripts

Thin bash wrappers, one per pipeline stage, with the paper settings filled in. Each script's header has its usage and an example; arguments after the positionals are forwarded to the Python entry point (except `run_humming_speedup.sh` and `push_to_hf.sh`); to forward extra arguments, fill in the optional positionals first. Run the scripts from the repository root.

```
scripts/run_gptq.sh                  <model_id>   <save_dir>     [num_gpus]
scripts/run_build_database.sh        <model_id>   <layer_dir>
scripts/run_search_dl.sh             <model_id>   <layer_dir>    [ear_target]
scripts/run_search_tl_calibrated.sh  <model_id>   <layer_dir>    <bf16_results.json> <anchor_results.json> [min_recovery] [cal_bitwidth]
scripts/run_search_tl.sh             <model_id>   <layer_dir>    [min_recovery] [num_gpus] [temperature] [top_p] [top_k]
scripts/run_build_checkpoint.sh      <model_id>   <layer_dir>    <config | bits>  <out_dir>
scripts/run_lmeval.sh                <checkpoint> <out_dir>      [num_gpus] [temperature] [top_p] [top_k]
scripts/run_ppl.sh                   <checkpoint> [datasets]
scripts/run_ear.sh                   <model_id>   <layer_dir>    <config.txt> ...
scripts/run_humming_speedup.sh       <model_id>   <layer_dir>    [batch] [seq_len] [gpu] [bf16_gpus]
scripts/push_to_hf.sh                <checkpoint> <hf_repo_id>   [--private]
```

`run_gptq.sh` writes the per-layer weights to `<save_dir>/<model_name>/4bit/` (4 is the calibration bitwidth); that directory is the `<layer_dir>` of every later stage.

`run_search_dl.sh` measures EAR with one forward pass per binary-search step; append `--use-predicted-constraints` for the purely predicted variant of paper Sec. 3.3.

`run_search_tl_calibrated.sh` is Algorithm 2 (single-point calibration + guardrail) and minimizes the KL Shapley costs. The anchor recovery is the mean per-benchmark score ratio between the two `run_lmeval.sh` results: one for the BF16 model and one for the uniform `cal_bitwidth` model from `<layer_dir>`, built with `run_build_checkpoint.sh <model_id> <layer_dir> <cal_bitwidth> <out_dir> --quant-format fake` (see the top-level README).

`run_search_tl.sh` is the benchmark-in-the-loop variant that runs the lm-eval suite at every binary-search step; use it for validation. It works in `<layer_dir>/search_tl_work` and keeps only the best candidate's BF16 fake-quant checkpoint (path in `search_history.json` under `best.checkpoint`).

Sampling defaults of `run_lmeval.sh` and `run_search_tl.sh` are Qwen3's (`0.6 0.95 20`). Use `1.0 0.95 20` for Qwen3.5 and `0.6 0.9 0` for Llama-3.3 (top_k 0 = off).

`run_ear.sh` reports EAR and top-K KL on the calibration set (512 x 2048 FineWeb-Edu tokens) for configs applied to the BF16 model; append e.g. `--uniform-bitwidths 4,8` after the config files for uniform baselines.

`run_humming_speedup.sh` times model-level forward passes for uniform 2..8-bit Humming models against BF16 (no direct paper figure), with `--single-gpu --compile` on the quantized runs. `gpu` (default 0) is used for the Humming runs; `bf16_gpus` may be a comma list for models whose BF16 weights do not fit on one GPU. Humming is taken from `$HUMMING_ROOT` (default `../humming`) if that directory exists, otherwise from the installed package.

`push_to_hf.sh` uploads a checkpoint built by `slq_build_checkpoint.py`, including its `serve.sh` launcher and model card, to the Hugging Face Hub.
