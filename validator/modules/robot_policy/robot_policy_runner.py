"""Real RobotPolicy runner: rolls out an exported ONNX policy in the official
MjLab/MuJoCo-Warp stack.

NOTHING here reimplements robot physics. The environment, the observation
construction and the reward/termination logic are all the official
`mjlab_microduck` task, instantiated by its registered task id. That matters
most for the OBSERVATION: the policy consumes a 61-D vector assembled in a
very specific term order, and hand-rolling it would silently produce a
different input distribution than training. We ask mjlab for `obs["actor"]`
and feed it straight to onnxruntime.

Requires the pinned upstream stack (mjlab_microduck, mjlab, torch,
mujoco-warp, onnxruntime) in the module's conda environment.

--------------------------------------------------------------------------
PERTURBATION MAPPING
--------------------------------------------------------------------------
The suite YAML speaks benchmark language; the simulator speaks BAM/mjlab
language. Every mapping below was verified against the installed source, and
each is applied WITHOUT rebuilding the environment (a rebuild per episode
would cost ~15 s x 100 episodes):

  friction (0.55 .. 1.0)
      -> absolute friction on the two foot collision geoms, applied by calling
         the task's own `foot_friction` event function with ranges=(f, f).
         NOTE this is a STARTUP event in the task, so it does not re-run on
         reset; we invoke it explicitly per scenario.

  command_delay_steps (0 .. 3, CONTROL steps)
      -> mjlab actuator delay, which is measured in PHYSICS timesteps.
         decimation is 4 (50 Hz control on a 200 Hz sim), so
             physics_lag = control_steps * decimation
         The DelayBuffer resamples its lag EVERY step from [min_lag, max_lag]
         (update_period=0), so setting lags alone would be overwritten on the
         next step -- we pin min_lag == max_lag == physics_lag instead.
         The buffer's ring size is fixed at construction to max_lag+1, so the
         env is built with enough headroom for the largest lag in the suite.

  battery_voltage (6.5 .. 7.5 V)
      -> BamActuator.vin_tensor, a live per-env (N,1) tensor. BAM samples it at
         startup and its reset() explicitly does NOT re-sample it, so a direct
         write is stable for the whole episode.

  DETERMINISM ACROSS PROCESSES
      The task randomizes per-env physics in FIVE startup events -- foot
      friction, encoder bias, base CoM, mass/inertia -- plus BAM's battery
      voltage and voltage-drop gain. All of them are sampled ONCE at env
      CONSTRUCTION, from the global torch RNG, long before any per-scenario
      seed is set. Two validator processes therefore instantiate physically
      DIFFERENT robots, and the same scenario diverges: measured 8.6 cm spread
      in final trunk height between two runs of the same policy on the same
      seed. A benchmark cannot do that, so the env is built under a fixed
      `build_seed` that belongs to the frozen suite definition.

  initial_states (face_down | face_up | sitting)
      -> the task's `set_ground_state` reset event, whose pose probabilities we
         set to a one-hot per scenario.
         Mutated through `env.event_manager.get_term_cfg(...)`, NEVER through
         `env.cfg.events[...]`: managers deepcopy their cfg at construction, so
         writes to env.cfg are silent no-ops. AGENTS.md calls this out
         specifically as something that "bites eval scripts that force spawn
         states".
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .substrate import SIMULATION

# Physics timesteps per control step (mjlab decimation). Verified at runtime
# against env.step_dt and the sim timestep; see _check_timing().
DEFAULT_DECIMATION = 4

# A sample is flagged `fallen` when the trunk is tilted past this angle.
# The success predicate applies its own (tighter) tilt limit; this flag exists
# so the adapter can distinguish "on the floor" from "upright but marginal".
FALLEN_TILT_DEG = 60.0


def _tilt_deg(quat_wxyz) -> float:
    """Angle between the trunk's body-up axis and world up, in degrees.

    quat is (w, x, y, z). The body z-axis in world frame is the third column of
    the rotation matrix; its dot with world up is R[2,2].
    """
    _, x, y, _ = (float(v) for v in quat_wxyz)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, r22))))


def simulator_stack() -> dict:
    """What actually ran the physics.

    A simulation result is only reproducible against a specific stack, and
    until this existed every result file recorded the simulator as the literal
    string "local". Two results could disagree because the policies differ,
    or because one was produced by a different mujoco-warp -- and nothing on
    disk could tell you which. Read from the IMPORTED modules rather than from
    a lockfile, so it reports what was really loaded.

    Also records the upstream checkout's git commit: mjlab is pinned by
    version, but the RobotPolicy tasks -- the reward terms, the observation
    order, the startup randomisation -- live in the checkout, and its commit
    is what decides whether two runs evaluated the same task at all.
    """
    import importlib
    import subprocess

    out = {}
    for mod in ("mjlab", "mjlab_microduck", "mujoco", "mujoco_warp",
                "warp", "torch", "onnxruntime", "numpy"):
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        v = getattr(m, "__version__", None)
        if v is None and mod == "warp":
            v = getattr(getattr(m, "config", None), "version", None)
        if v is None:
            # mjlab ships no __version__; fall back to installed metadata.
            try:
                from importlib.metadata import version as _pv
                v = _pv(mod)
            except Exception:
                v = "unknown"
        out[mod] = str(v)
    # The checkout the tasks came from.
    try:
        import mjlab_microduck
        repo = Path(mjlab_microduck.__file__).resolve().parents[2]
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            commit = r.stdout.strip()
            # --untracked-files=no on purpose: a stray note or log sitting in
            # the checkout does not change what the simulator did. Only
            # MODIFIED TRACKED files mean the commit no longer describes the
            # code that ran, and that is worth flagging loudly.
            d = subprocess.run(["git", "-C", str(repo), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10)
            out["microduck_rl_commit"] = commit + ("-dirty" if d.stdout.strip() else "")
            out["microduck_rl_path"] = str(repo)
    except Exception:
        pass
    return out


class RobotPolicyRunner:
    """Rollout backend backed by the official mjlab task. SIMULATION substrate."""

    substrate = SIMULATION

    def __init__(self, task_id: str | None = None, device: str = "cuda:0",
                 max_delay_control_steps: int = 3,
                 decimation: int = DEFAULT_DECIMATION,
                 fallen_tilt_deg: float = FALLEN_TILT_DEG,
                 build_seed: int = 0, onnx_provider: str = "cpu",
                 live_path: str | None = None, live_hz: float = 15.0,
                 batch: int = 1):
        # Scenarios simulated at once. 1 is the original one-env sequential
        # runner; N runs N scenarios as N parallel envs of one mjlab instance
        # (rollout_batch). Borrowed from DuckEMW's eval_sprint_speed.py.
        self.batch = max(1, int(batch))
        self._batch_ok = None          # does this ONNX accept a batch dim > 1?
        self._last_batch = 1
        # The policy is a 61->14 MLP: ~50k MACs per step. On CPU that is
        # microseconds, and the physics step dominates by orders of magnitude,
        # so CUDA execution is OFF by default -- it would add a host<->device
        # copy per step and contend with the GPU already running the sim.
        # Offered because it can help when a future policy is much larger.
        self.onnx_provider = onnx_provider
        self.build_seed = build_seed
        # Optional live pose stream, in the same wire format the trainer uses,
        # so the control room draws a validation episode in the same official
        # scene it draws training in. Costs one device sync per published
        # frame; measured at +0.2% on the training loop, and a validation
        # rollout is a single environment.
        self.live_path = Path(live_path) if live_path else None
        self.live_period = 1.0 / max(1e-3, float(live_hz))
        self._live_last = 0.0
        self._live_perm = None
        self.task_id = task_id
        self.device = device
        self.decimation = decimation
        self.fallen_tilt_deg = fallen_tilt_deg
        self.max_delay_control_steps = max_delay_control_steps
        self._env = None
        self._sess = None
        self._input_name = None
        self._policy_path = None
        self._built_for = None

    # ---------------------------------------------------------------- build
    def _task_config(self, manifest) -> dict:
        """The task config the manifest points at (predicate, envelope), cached."""
        path = manifest.resolve(manifest.raw["task"]["config"])
        if getattr(self, "_task_cfg_path", None) != path:
            import yaml
            self._task_cfg = yaml.safe_load(Path(path).read_text()) or {}
            self._task_cfg_path = path
        return self._task_cfg

    @staticmethod
    def _twist_cfg(cfg):
        cmds = getattr(cfg, "commands", None)
        if cmds is None:
            return None
        return cmds.get("twist") if isinstance(cmds, dict) else getattr(cmds, "twist", None)

    def _build(self, task_id: str, timeout_s: float, envelope=None, one_cmd=False, num_envs=1):
        import numpy as np
        import torch
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        # MUST precede construction: the five startup randomization events and
        # BAM's vin_tensor draw from the global RNG at build time. Without this
        # every process gets a different robot and the benchmark is not
        # reproducible across validators. See the module docstring.
        torch.manual_seed(self.build_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.build_seed)
        np.random.seed(self.build_seed)

        # play=True gives the evaluation variant of the task (training-only
        # curricula/noise off) while keeping the physics and obs identical.
        cfg = load_env_cfg(task_id, play=True)
        cfg.scene.num_envs = int(num_envs)
        # Episodes must be able to run the full benchmark timeout without the
        # env truncating underneath us.
        if hasattr(cfg, "episode_length_s"):
            cfg.episode_length_s = float(timeout_s) + 1.0
        # Commands. By default the env draws twists from its TRAINING ranges
        # (vx +-0.4, vy +-0.3 for the velocity task); the deployed runtime never
        # commands lateral motion and caps vx at 0.25/-0.2, and a lateral 0.3
        # made Pollen's own policy fall. A task config may pin the benchmark to
        # the deployment envelope (`command_envelope`), and hold one command for
        # the whole episode so a windowed tracking metric is not fooled by the
        # env's own 3-8 s resample landing mid-episode.
        twist = self._twist_cfg(cfg)
        if envelope:
            if not isinstance(envelope, dict):
                raise RuntimeError("configured command envelope must be a mapping")
            if twist is None or getattr(twist, "ranges", None) is None:
                raise RuntimeError("simulator cannot apply the configured command envelope")
            updates = {}
            for k, v in envelope.items():
                if k not in ("lin_vel_x", "lin_vel_y", "ang_vel_z") or not hasattr(twist.ranges, k):
                    raise RuntimeError("simulator does not support a configured command range")
                if (not isinstance(v, (list, tuple)) or len(v) != 2
                        or any(type(x) not in (int, float) or not math.isfinite(x) for x in v)
                        or v[0] > v[1]):
                    raise RuntimeError("configured command ranges must have finite ordered bounds")
                updates[k] = (float(v[0]), float(v[1]))
            for k, v in updates.items():
                setattr(twist.ranges, k, v)
        if one_cmd:
            if twist is None or not hasattr(twist, "resampling_time_range"):
                raise RuntimeError("simulator cannot hold one command for the full episode")
            twist.resampling_time_range = (float(timeout_s) + 2.0, float(timeout_s) + 2.0)

        # Give the delay ring buffer room for the largest lag the suite can ask
        # for; its size is fixed at construction (max_lag + 1).
        max_phys_lag = self.max_delay_control_steps * self.decimation
        for ent in (cfg.scene.entities.values() if hasattr(cfg.scene, "entities") else []):
            art = getattr(ent, "articulation", None)
            for act in (getattr(art, "actuators", None) or []):
                if hasattr(act, "delay_max_lag"):
                    act.delay_min_lag = 0
                    act.delay_max_lag = max(int(getattr(act, "delay_max_lag", 0)),
                                            max_phys_lag)

        env = ManagerBasedRlEnv(cfg=cfg, device=self.device)
        self._env = env
        # The delay ring is sized ONCE, from max_lag at construction. Remember
        # that size: _apply_delay used to cap the requested lag with the
        # CURRENT max_lag, which it had itself just shrunk to the previous
        # scenario's lag -- so after one zero-delay scenario every later
        # command_delay_steps silently became 0. Found while writing the
        # batched runner, 2026-09-05.
        for act in self._actuators():
            buf = getattr(act, "_delay_buffer", None)
            if buf is not None and not hasattr(buf, "_ring_max"):
                buf._ring_max = int(buf.max_lag)
        self._check_timing(env)
        return env

    def _check_timing(self, env) -> None:
        """Fail loudly if control rate is not the 50 Hz the suite assumes."""
        dt = float(env.step_dt)
        if abs(dt - 0.02) > 1e-6:
            raise RuntimeError(
                f"env.step_dt = {dt} s, but the suite's sample_dt_s is 0.02 s "
                "(50 Hz). The benchmark trajectory sampling rate and the policy "
                "control rate must match."
            )

    def _load_policy(self, path: Path):
        import onnxruntime as ort
        if self._policy_path == path and self._sess is not None:
            return
        self._batch_ok = None
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if self.onnx_provider == "cuda" else ["CPUExecutionProvider"])
        try:
            sess = ort.InferenceSession(str(path), providers=providers)
        except Exception:
            if self.onnx_provider != "cuda":
                raise
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        from .policy import check_session_contract
        check_session_contract(sess)
        inp = sess.get_inputs()[0]
        self._sess, self._input_name, self._policy_path = sess, inp.name, path

    # ------------------------------------------------------- perturbations
    def _force_initial_state(self, initial: str) -> None:
        """One-hot the spawn-pose probabilities of `set_ground_state`."""
        env = self._env
        if "set_ground_state" not in (env.event_manager.active_terms.get("reset") or []):
            # Locomotion tasks spawn standing and have no pose-choice event.
            # That is only compatible with a suite asking for "standing".
            if initial != "standing":
                raise RuntimeError(
                    f"suite asks for initial state {initial!r} but task "
                    f"{self.task_id} has no 'set_ground_state' event to force it")
            return
        term = env.event_manager.get_term_cfg("set_ground_state")
        keys = {"face_down": "face_down_prob", "face_up": "face_up_prob",
                "sitting": "sitting_prob", "standing": "standing_prob"}
        if initial not in keys:
            raise ValueError(f"unsupported initial state {initial!r}")
        for name in keys.values():
            if name in term.params:
                term.params[name] = 0.0
        term.params[keys[initial]] = 1.0

    def _apply_friction(self, friction: float, env_ids) -> None:
        env = self._env
        try:
            term = env.event_manager.get_term_cfg("foot_friction")
            if not callable(getattr(term, "func", None)):
                raise RuntimeError("foot friction event is unavailable")
            params = dict(term.params)
            params["ranges"] = (float(friction), float(friction))
            params["operation"] = "abs"
            term.func(env, env_ids, **params)
        except Exception as exc:
            raise RuntimeError("simulator could not apply the requested foot friction") from exc

    def _actuators(self):
        robot = self._env.scene["robot"]
        return list(getattr(robot, "actuators", []) or [])

    def _apply_delay(self, control_steps: int) -> None:
        self._apply_delays([control_steps] * int(self._env.num_envs))

    def _apply_delays(self, control_steps) -> None:
        """Per-env command delay, in control steps (one entry per env)."""
        import torch
        if (len(control_steps) != int(self._env.num_envs) or not control_steps
                or any(type(value) is not int or value < 0 for value in control_steps)
                or type(self.decimation) is not int or self.decimation <= 0):
            raise RuntimeError("command delay requires one nonnegative integer per environment")
        # This buffer re-samples from a shared min/max interval each step.
        # Distinct per-env lags would therefore be overwritten after set_lags.
        if len(set(control_steps)) > 1:
            raise RuntimeError("mixed command delays cannot be held by this simulator; use batch=1 for this suite")
        phys = [value * self.decimation for value in control_steps]
        targets = []
        for act in self._actuators():
            buf = getattr(act, "_delay_buffer", None)
            if buf is None:
                continue
            ring = int(getattr(buf, "_ring_max", buf.max_lag))   # size fixed at construction
            if max(phys) > ring:
                raise RuntimeError("requested command delay exceeds the simulator buffer capacity")
            if not callable(getattr(act, "set_lags", None)):
                raise RuntimeError("simulator actuator cannot apply the requested command delay")
            targets.append((act, buf))
        if not targets:
            raise RuntimeError("simulator has no actuator supporting command delay")
        for act, buf in targets:
            buf.min_lag = min(phys)
            buf.max_lag = max(phys)
            try:
                act.set_lags(torch.tensor(phys, dtype=torch.long, device=self._env.device))
            except Exception as exc:
                raise RuntimeError("simulator could not apply the requested command delay") from exc

    def _apply_voltage(self, volts: float) -> None:
        self._apply_voltages([volts] * int(self._env.num_envs))

    def _apply_voltages(self, volts) -> None:
        """Per-env battery voltage (None = leave that env at its default)."""
        n = int(self._env.num_envs)
        if len(volts) != n:
            raise RuntimeError("battery voltage requires one value per environment")
        if all(value is None for value in volts):
            return
        if any(value is not None and (type(value) not in (int, float)
                                      or not math.isfinite(value) or value <= 0)
               for value in volts):
            raise RuntimeError("battery voltages must be positive finite values")
        targets = []
        for act in self._actuators():
            vt = getattr(act, "vin_tensor", None)
            if vt is None:
                continue
            shape = getattr(vt, "shape", ())
            if not shape or shape[0] != n:
                raise RuntimeError("simulator battery voltage is not independently addressable per environment")
            targets.append(vt)
        if not targets:
            raise RuntimeError("simulator has no actuator supporting battery voltage")
        for vt in targets:
            for j, v in enumerate(volts):
                if v is None:
                    continue
                try:
                    vt[j] = float(v)
                except Exception as exc:
                    raise RuntimeError("simulator could not apply per-environment battery voltage") from exc

    def _infer(self, a):
        import numpy as np
        from .errors import InvalidSubmission
        if a.ndim != 2 or a.shape[1] != 61 or not np.isfinite(a).all():
            raise RuntimeError("simulator produced invalid observations")
        out = self._infer_raw(a)
        if out.shape != (a.shape[0], 14) or not np.isfinite(out).all():
            raise InvalidSubmission("off-contract: policy actions must be finite [batch, 14]")
        return out

    def _infer_raw(self, a):
        """Run the ONNX on (n, 61). Exports carry a fixed batch of 1; find out
        once whether this one takes a batch, else loop the rows."""
        import numpy as np
        if a.shape[0] == 1:
            return self._sess.run(None, {self._input_name: a})[0]
        if self._batch_ok is None:
            try:
                out = self._sess.run(None, {self._input_name: a})[0]
                self._batch_ok = out.shape[0] == a.shape[0]
                if self._batch_ok:
                    return out
            except Exception:  # noqa: BLE001 -- fixed-batch export
                self._batch_ok = False
        if self._batch_ok:
            return self._sess.run(None, {self._input_name: a})[0]
        return np.concatenate([self._sess.run(None, {self._input_name: a[i:i + 1]})[0]
                               for i in range(a.shape[0])], axis=0)
    _LIVE_JOINTS = (
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "neck_pitch", "head_pitch", "head_yaw", "head_roll",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
        "right_ankle",
    )

    def _publish_live(self, env, robot, meta: dict) -> None:
        """One env's pose, rate-limited, written atomically."""
        import time as _t

        if self.live_path is None:
            return
        now = _t.monotonic()
        if now - self._live_last < self.live_period:
            return
        self._live_last = now
        try:
            if self._live_perm is None:
                names = list(getattr(robot, "joint_names", None)
                             or getattr(robot.data, "joint_names", []))
                missing = [n for n in self._LIVE_JOINTS if n not in names]
                if missing:
                    self.live_path = None      # not this robot; stop trying
                    return
                self._live_perm = [names.index(n) for n in self._LIVE_JOINTS]
            import torch
            d = robot.data
            k = min(int(env.num_envs), 36)         # the swarm's own cap
            # One device sync for everything, all envs at once.
            cols = [d.root_link_pos_w[:k] - env.scene.env_origins[:k],
                    d.root_link_quat_w[:k], d.joint_pos[:k]]
            ball = None
            try:                                   # tasks with an object in the scene
                ball = env.scene["ball"]
                cols.append(ball.data.root_link_pos_w[:k] - env.scene.env_origins[:k])
            except (KeyError, TypeError, AttributeError):
                ball = None
            rows = torch.cat(cols, dim=1).detach().cpu().numpy()
            nj = int(d.joint_pos.shape[1])
            r3 = lambda x: round(float(x), 3)
            ents = []
            for row in rows:
                ent = {"p": [r3(row[i]) for i in range(7)],
                       "j": [r3(row[7 + i]) for i in self._live_perm]}
                if ball is not None:
                    ent["b"] = [r3(row[7 + nj]), r3(row[8 + nj]), r3(row[9 + nj])]
                ents.append(ent)
            frame = {
                "t": _t.time(), "n": len(ents), "hz": round(1.0 / self.live_period, 1),
                "num_envs": int(env.num_envs), "source": "validation", **meta,
                "envs": ents,
            }
            tmp = self.live_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(frame, separators=(",", ":")))
            tmp.replace(self.live_path)
        except Exception:
            # A live view is never worth losing a benchmark run over.
            self.live_path = None

    def rollout(self, case, manifest):
        import numpy as np
        import torch

        task_id = self.task_id or manifest.raw["task"]["id"]
        timeout = float(case["timeout"])
        task_cfg = self._task_config(manifest)
        envelope = task_cfg.get("command_envelope") or None
        one_cmd = bool(task_cfg.get("one_command_per_episode", False))
        key = (task_id, round(timeout, 3), json.dumps(envelope, sort_keys=True), one_cmd)
        if self._env is None or self._built_for != key:
            if self._env is not None:
                self._env.close()
                self._env = None
            self._build(task_id, timeout, envelope, one_cmd)
            self._built_for = key
        env = self._env
        self._load_policy(Path(manifest.policy))
        self._last_batch = 1

        pert = case.get("perturbations", {}) or {}
        seed = int(case["seed"])

        # Deterministic per-scenario reset.
        self._force_initial_state(case["initial"])
        torch.manual_seed(seed)
        np.random.seed(seed % (2**31 - 1))
        obs, _ = env.reset(seed=seed)

        env_ids = torch.zeros(1, dtype=torch.long, device=env.device)
        if "friction" in pert:
            self._apply_friction(pert["friction"], env_ids)
        if "command_delay_steps" in pert:
            self._apply_delay(int(pert["command_delay_steps"]))
        if "battery_voltage" in pert:
            self._apply_voltage(pert["battery_voltage"])

        robot = env.scene["robot"]
        # Emit a SUPERSET of canonical fields, not just the ones the current
        # task family needs: adapters pick what they use, and a new family
        # costs no runner change. `twist` exists on every locomotion task; the
        # standup family simply ignores these columns.
        has_twist = "twist" in getattr(env.command_manager, "active_terms", []) or []
        n_steps = int(round(timeout / float(env.step_dt)))
        traj = []
        terminated_at = None
        terminated_by = []

        for i in range(n_steps):
            a = obs["actor"].detach().cpu().numpy().astype(np.float32)
            act = self._infer(a)
            obs, _rew, term, trunc, _info = env.step(
                torch.as_tensor(act, dtype=torch.float32, device=env.device))

            self._publish_live(env, robot, {
                "scenario": case.get("id") or case.get("scenario_id"),
                "step": i, "steps": n_steps,
                "policy": Path(manifest.policy).name,
            })

            z = float(robot.data.root_link_pos_w[0, 2])
            tilt = _tilt_deg(robot.data.root_link_quat_w[0].tolist())
            sample = {
                "t": round((i + 1) * float(env.step_dt), 4),
                "base_height_m": z,
                "tilt_deg": tilt,
                "fallen": bool(tilt > self.fallen_tilt_deg),
            }
            if has_twist:
                lv = robot.data.root_link_lin_vel_b[0]
                av = robot.data.root_link_ang_vel_b[0]
                cmd = env.command_manager.get_command("twist")[0]
                sample.update(
                    lin_vel_x=float(lv[0]), lin_vel_y=float(lv[1]),
                    ang_vel_z=float(av[2]),
                    cmd_lin_vel_x=float(cmd[0]), cmd_lin_vel_y=float(cmd[1]),
                    cmd_ang_vel_z=float(cmd[2]))
            if bool(term[0]) or bool(trunc[0]):
                # mjlab resets a terminated env INSIDE step(), so everything
                # read from `robot.data` after this step is the fresh reset
                # pose: upright, at rest, on the spawn cell. Recording it as
                # the episode's last sample made every fall look like a robot
                # standing still with 0.0 deg of tilt -- and it did, for 100
                # episodes, until someone asked why the untrained policy's
                # episodes were 1 s long. Keep the time and the reason; the
                # terminal state itself is not observable.
                terminated_at = sample["t"]
                tm = env.termination_manager
                terminated_by = [n for n in tm.active_terms
                                 if bool(tm.get_term(n)[0])]
                if "fell_over" in terminated_by and traj:
                    traj[-1]["fallen"] = True
                break
            traj.append(sample)

        return traj, self._evidence(task_id, case, pert, traj, terminated_at, terminated_by,
                                    manifest, has_twist, env)

    def _evidence(self, task_id, case, pert, traj, terminated_at, terminated_by,
                  manifest, has_twist, env):
        evidence = {
            "runner": "robot_policy",
            "task_id": task_id,
            "device": self.device,
            "policy": Path(manifest.policy).name,
            "control_hz": round(1.0 / float(env.step_dt), 3),
            "build_seed": self.build_seed,
            "onnx_provider": self.onnx_provider,
            "batch": self._last_batch,
            "steps": len(traj),
            "initial_state": case["initial"],
            "perturbations_applied": {
                "friction": pert.get("friction"),
                "command_delay_steps": pert.get("command_delay_steps"),
                "command_delay_physics_steps":
                    None if "command_delay_steps" not in pert
                    else int(pert["command_delay_steps"]) * self.decimation,
                "battery_voltage": pert.get("battery_voltage"),
            },
            "env_terminated_at_s": terminated_at,
            "env_terminated_by": terminated_by,
            "final_base_height_m": traj[-1]["base_height_m"] if traj else None,
            "final_tilt_deg": traj[-1]["tilt_deg"] if traj else None,
        }
        if has_twist and traj:
            # Same definition as the adapter's verdict (window, weights), so
            # the reported error and the pass/fail agree with each other.
            from .adapters import velocity_errors
            pred = self._task_config(manifest).get("success_predicate") or {}
            errs, mags = velocity_errors(traj, pred)
            evidence["mean_tracking_error"] = sum(errs) / len(errs)
            # How much motion was actually ASKED FOR: raw tracking error is
            # confounded by it (see configs/tasks/TEMPLATE.yaml, cmd_magnitude_floor).
            evidence["mean_cmd_magnitude"] = sum(mags) / len(mags)
            evidence["command_envelope"] = self._task_config(manifest).get("command_envelope")
        return evidence

    def rollout_batch(self, cases, manifest):
        """Simulate `cases` as parallel envs of one mjlab instance.

        Same physics, same per-scenario perturbations (friction, command delay
        and battery voltage applied PER ENV), same trajectory format as
        rollout(). What differs: the env's own reset randomisation is drawn
        from one RNG stream seeded from all the scenarios' seeds together, so
        an episode is reproducible given the suite and the batch layout, not
        bit-identical to the sequential runner's episode for that scenario.
        On warp GPU the sequential runner is not bit-identical to itself
        either (AGENTS.md). Every case in a chunk shares `initial` and
        `timeout`; the engine groups them that way.
        """
        import hashlib
        import numpy as np
        import torch
        if not cases:
            return []
        task_id = self.task_id or manifest.raw["task"]["id"]
        timeout = float(cases[0]["timeout"])
        initial = cases[0]["initial"]
        for c in cases:
            if abs(float(c["timeout"]) - timeout) > 1e-9 or c["initial"] != initial:
                raise ValueError("rollout_batch: every case in a chunk must share timeout and initial state")
        n = len(cases)
        task_cfg = self._task_config(manifest)
        envelope = task_cfg.get("command_envelope") or None
        one_cmd = bool(task_cfg.get("one_command_per_episode", False))
        key = (task_id, round(timeout, 3), json.dumps(envelope, sort_keys=True), one_cmd, n)
        if self._env is None or self._built_for != key:
            if self._env is not None:
                self._env.close()
                self._env = None
            self._build(task_id, timeout, envelope, one_cmd, num_envs=n)
            self._built_for = key
        env = self._env
        self._load_policy(Path(manifest.policy))
        self._last_batch = n
        self._force_initial_state(initial)
        seeds = [int(c["seed"]) for c in cases]
        bseed = int(hashlib.sha256(",".join(map(str, seeds)).encode()).hexdigest()[:8], 16)
        torch.manual_seed(bseed)
        np.random.seed(bseed % (2**31 - 1))
        obs, _ = env.reset(seed=bseed)
        dev = env.device
        perts = [c.get("perturbations", {}) or {} for c in cases]
        by_friction: dict = {}
        for j, pt in enumerate(perts):
            if "friction" in pt:
                by_friction.setdefault(float(pt["friction"]), []).append(j)
        for f, ids in by_friction.items():
            self._apply_friction(f, torch.tensor(ids, dtype=torch.long, device=dev))
        if any("command_delay_steps" in pt for pt in perts):
            self._apply_delays([int(pt.get("command_delay_steps", 0)) for pt in perts])
        if any("battery_voltage" in pt for pt in perts):
            self._apply_voltages([pt.get("battery_voltage") for pt in perts])

        robot = env.scene["robot"]
        has_twist = "twist" in getattr(env.command_manager, "active_terms", []) or []
        n_steps = int(round(timeout / float(env.step_dt)))
        trajs = [[] for _ in range(n)]
        alive = [True] * n
        term_at = [None] * n
        term_by = [[] for _ in range(n)]
        tm = env.termination_manager
        for i in range(n_steps):
            a = obs["actor"].detach().cpu().numpy().astype(np.float32)
            act = self._infer(a)
            obs, _rew, term, trunc, _info = env.step(
                torch.as_tensor(act, dtype=torch.float32, device=dev))
            self._publish_live(env, robot, {
                "scenario": cases[0].get("id") or cases[0].get("scenario_id"),
                "step": i, "steps": n_steps, "policy": Path(manifest.policy).name,
            })
            d = robot.data
            pos = d.root_link_pos_w.detach().cpu().numpy()
            quat = d.root_link_quat_w.detach().cpu().numpy()
            if has_twist:
                lv = d.root_link_lin_vel_b.detach().cpu().numpy()
                av = d.root_link_ang_vel_b.detach().cpu().numpy()
                cmd = env.command_manager.get_command("twist").detach().cpu().numpy()
            ended = (term | trunc).detach().cpu().numpy()
            fired = ({name: tm.get_term(name).detach().cpu().numpy() for name in tm.active_terms}
                     if ended.any() else None)
            t = round((i + 1) * float(env.step_dt), 4)
            for j in range(n):
                if not alive[j]:
                    continue
                if ended[j]:
                    # mjlab has already reset this env inside step(): what is in
                    # robot.data for it now is the spawn pose. Keep the time and
                    # the reason, drop the sample -- exactly as rollout() does.
                    term_at[j] = t
                    term_by[j] = [nm for nm, arr in fired.items() if bool(arr[j])]
                    if "fell_over" in term_by[j] and trajs[j]:
                        trajs[j][-1]["fallen"] = True
                    alive[j] = False
                    continue
                tilt = _tilt_deg(quat[j].tolist())
                sample = {"t": t, "base_height_m": float(pos[j, 2]), "tilt_deg": tilt,
                          "fallen": bool(tilt > self.fallen_tilt_deg)}
                if has_twist:
                    sample.update(lin_vel_x=float(lv[j, 0]), lin_vel_y=float(lv[j, 1]),
                                  ang_vel_z=float(av[j, 2]),
                                  cmd_lin_vel_x=float(cmd[j, 0]), cmd_lin_vel_y=float(cmd[j, 1]),
                                  cmd_ang_vel_z=float(cmd[j, 2]))
                trajs[j].append(sample)
            if not any(alive):
                break
        return [(trajs[j], self._evidence(task_id, cases[j], perts[j], trajs[j], term_at[j],
                                          term_by[j], manifest, has_twist, env))
                for j in range(n)]

    def describe(self):
        """Identity fields the consensus layer uses as independence axes."""
        return {"substrate": self.substrate, "device": self.device,
                "build_seed": self.build_seed,
                "onnx_provider": self.onnx_provider,
                "batch": self.batch,
                "simulator_stack": simulator_stack()}

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
