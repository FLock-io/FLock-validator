# robot_policy validation contract

RobotPolicy uses the existing `ValidationRunner` and FedLedger request/result
endpoints. FedLedger selects the dataset for each assignment:

The module name and FedLedger task type are `robot_policy`. Configuration is loaded
from `configs/robot_policy.json`; evaluation output defaults to `.cache/robot_policy`.
Start validation with `python run.py robot_policy --task_ids <task_id>` after setting
`FLOCK_API_KEY` and `HF_TOKEN`. Python callers import from
`validator.modules.robot_policy` and use the `RobotPolicyConfig`, `RobotPolicyInputData`,
`RobotPolicyMetrics` and `RobotPolicyValidationModule` classes.

The upstream `mjlab_microduck` package, its registered `Mjlab-*-MicroDuck` task
IDs and robot embodiment metadata retain their simulator-defined names.

| Server phase | Server dataset key | Validator input |
| --- | --- | --- |
| `training` | `validation_set_key` | `data.validation_set_url` |
| `final_validation` | `final_validation_set_key` | `data.validation_set_url` |

The validator downloads a **suite YAML**, whereas the RL module downloads an
NPZ dataset. There is no local `final_suite` switch and no fallback dataset.
Remove the former `suite` setting from module/task JSON configurations.
Local configuration continues to own the task, scoring rules and runtime.

## Assignment boundary

RobotPolicy uses the unchanged runner: merge `task_submission.data` with
assignment `data`, with assignment values taking precedence, then validate
the merged input. The input fields are the same five fields used by RL.
No extra assignment metadata is required by the module.

- `task_submission.data` provides `hg_repo_id`, `model_filename`, `revision`.
  The revision must be a full 40-character hexadecimal HF commit ID.
- Assignment `data` provides a required HTTPS `validation_set_url` and a
  positive integer `max_params`.
- The existing merge preserves submission values when assignment `data` has
  no corresponding key. Unused input fields are ignored, as in the RL module.
- The runner uses the existing top-level `id` to submit results or mark a
  failed model. There is no additional `assignment_id` input or API parameter.
- Each evaluation has a separate local output directory. The module does not
  require a top-level `validation_phase`; FedLedger selects the dataset URL.

## Suite format

This is a public **training example**, not a private final test set:

```yaml
suite_id: velocity-training-v1.0
suite_family: velocity-standard-v0.1
suite_type: standard
mjlab_task_id: Mjlab-Velocity-Flat-MicroDuck
validation_phase: training
episodes: 100
suite_seed: 20260904
timeout_s: 5.0
scenario_distribution:
  initial_states: [standing]
  perturbations:
    friction: {min: 0.85, max: 1.0}
    command_delay_steps: [0, 1]
```

For a final dataset, assign a new `suite_id` and privately generate a different fixed
`suite_seed`. Freeze its episode count and distribution before evaluation.
Do not publish the private seed or commit the final dataset to this repository.
Increasing `episodes` with the same seed would repeat the earlier scenarios.
The existing YAMLs under `configs/suites` remain local benchmark references;
an online dataset also requires the `mjlab_task_id` field above.
`validation_phase` is optional descriptive suite metadata; when present, it
must be `training` or `final_validation` and is copied to the local result.
It is not used to select the dataset or required in the assignment response.

The parser rejects unknown or duplicate keys, YAML aliases, missing fields,
non-finite numbers, and booleans/strings in place of integers. It enforces:

- `suite_id` and optional `suite_family`: safe identifiers, at most 128 characters.
- `suite_type`: `standard` or `robustness`; it does not select the server phase.
- `mjlab_task_id`: an exact match to the local task. The supported tasks are
  the existing velocity and standup environments from the upstream simulator.
- `episodes`: 1–10,000; `suite_seed`: unsigned 64-bit integer.
- `timeout_s`: positive, at most 300 seconds, a whole number of 50 Hz steps.
- Initial states must be supported by the selected task.
- Friction: positive and at most 5; battery voltage: positive and at most 20.
  These accept a scalar, nonempty list, or ordered `{min, max}` range.
- Command delay: integer scalar or nonempty integer list, 0–3 control steps.
  Continuous ranges are rejected because the engine samples them as floats.

Suite downloads require HTTPS, HTTP 200 and UTF-8 YAML. They reject credentials
in URLs, fragments, localhost/non-public literal IP addresses and redirects.
Downloads have connection/read timeouts, a total deadline and a 1 MiB limit.
DNS hostnames are trusted as server-owned inputs; these URL checks are not a
DNS-based network egress allowlist. Signed URLs are not written to results or
included in downloader error messages. A file is saved atomically only after
its complete contents have passed validation.

## Evaluation and failure handling

The sequence follows RL: validate assignment → fetch the pinned model → check
model contract and parameter budget → download the selected suite → evaluate
all scenarios → check result consistency → submit `data.average_reward`.

- An oversized model returns the existing RL penalty, `-999`, without running
  the suite. Invalid ONNX contracts follow the existing failed-model path.
- Dataset/network/phase/task/runtime errors are recoverable. They do not fail
  the submitted model or produce a substitute score. Restart after fixing the
  underlying issue, following the existing runner's recovery behavior.
- Partial/empty results, non-finite metrics, inconsistent success counts,
  wrong scenario IDs and mismatched suite/task/model hashes cannot
  be submitted. The simulator is closed on both successful and failed runs.
- A result-submission HTTP error propagates; it is not reported as success or
  converted to a model failure. This preserves the existing runner behavior.

Use the same pinned physics stack, task/scoring config, `build_seed` and `batch`
on every validator. The default `batch=1` supports the example's varying delays.
The current delay buffer cannot hold different delays within one batch; such a
batch is rejected instead of resampling a different benchmark. Missing runtime
support for an explicitly requested perturbation also causes an error.

## Runtime and backend prerequisites

`environment.yml` installs Python 3.12 and the complete runtime from
`requirements-runtime.txt`, including the upstream robot assets and BAM actuator
model. The task and actuator sources are pinned to full commits; mjlab, MuJoCo,
MuJoCo-Warp, Warp, PyTorch and ONNX versions are fixed together. Do not upgrade
one physics package independently. Use `HF_TOKEN` for private model access.

From the repository root, on Linux or inside WSL2 with an NVIDIA GPU:

```bash
conda env create -n flock-validation-robot_policy -f validator/modules/robot_policy/environment.yml
conda run --no-capture-output -n flock-validation-robot_policy python validator/modules/robot_policy/check_runtime.py
conda run --no-capture-output -n flock-validation-robot_policy python validator/modules/robot_policy/call_validator_test.py
```

On WSL, if an old Linux NVIDIA driver is also installed, first run
`export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"`.
Warp opens `libcuda.so`, which can otherwise resolve to that old driver even
when PyTorch's `libcuda.so.1` resolves to the current Windows host driver.
The bootstrap below sets this process-local search path automatically; it does
not remove or replace system driver libraries.

If that environment already exists, use `conda env update` instead of `create`.
The ordinary `python run.py robot_policy ...` entry point creates/updates this
same Conda environment automatically. Directly running `call_validator_test.py`
uses the selected Python interpreter; it does not activate Conda for you.

Without Conda, the bootstrap script installs uv, Python 3.12 and an isolated
environment under `~/.cache/flock-robot-policy` on the Linux filesystem:

```bash
bash validator/modules/robot_policy/setup_runtime.sh --test
```

It needs `python3` with pip, network access for package/model downloads, and
several GB of free disk space. The uv package cache uses
`~/.cache/flock-robot-policy/uv` (or `UV_CACHE_DIR`). Keep both the cache and
environment on the Linux filesystem in WSL: Windows mounts can cause atomic
directory renames to fail with `Permission denied` and make simulator imports
very slow. Set `ROBOT_POLICY_RUNTIME_DIR` to choose another Linux directory.
On Windows, run it inside your WSL2 distribution,
for example from PowerShell at the repository root:

```powershell
wsl -d Ubuntu-20.04 -- bash validator/modules/robot_policy/setup_runtime.sh --test
```

For subsequent runs inside Linux/WSL, use
`~/.cache/flock-robot-policy/venv/bin/python validator/modules/robot_policy/call_validator_test.py`.
On WSL, retain the `LD_LIBRARY_PATH` setting above, or rerun the bootstrap with
`--test` to apply it automatically.
The sample reads `.env`, downloads the pinned ONNX model, and runs the local
three-episode suite on `cuda:0`; it does not submit a result to FedLedger.
`check_runtime.py` verifies imports, task registration and CUDA access, while
`call_validator_test.py` verifies actual simulation and scoring. A successful
dependency install alone is not a successful rollout. Native Windows simulation
is not the tested installation path; use Linux/WSL2 for this stack.

Verified on 2026-09-21 in WSL2 Ubuntu 20.04, Python 3.12.12, with an RTX 4090
Laptop GPU and host driver 528.66: dependency checks and CUDA checks passed;
`call_validator_test.py` exited successfully after all three local episodes
(250 control steps each), with `average_reward=0.0` and no partial suite.
This is a
runtime smoke test; the zero score records the policy's outcome on those three
scenarios. The upstream runtime automatically disables CUDA Graphs on this
older driver, while continuing to run the physics on CUDA.

FedLedger must register the `robot_policy` task type and serve its ordinary/final YAML
objects through the existing assignment fields. Its dataset preprocessing must
accept these YAML suites rather than invoke RL's NPZ preprocessing. This
repository implements the validator, not the FedLedger service or its database
migrations. A deployed backend must be verified separately before live use.

## Offline verification

With the module's non-physics dependencies, project requirements, `pytest`,
`onnx` and `onnxruntime` installed:

```bash
python -m pytest tests -q
```

Tests cover actual suite parsing, engine scoring and the existing ledger request
and submission bodies, including a cached module processing
`training → final_validation → training`. HTTP, HF model retrieval and the
physics boundary are replaced with controlled fixtures. RL uses its actual
NPZ loader and `EnvLite` in the regression tests; LoRA's existing input contract
is also covered. These tests do not establish GPU reproducibility, real policy
performance, or compatibility with a deployed FedLedger instance.
