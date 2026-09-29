from __future__ import annotations

import argparse
import ctypes
import errno
import os
import platform
import resource
import sys
from pathlib import Path
from typing import Any

import numpy as np


# Running this file directly puts robotics_vla/ on sys.path, not the repository
# root. Add the root explicitly before importing trusted validator modules.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validator.modules.robotics_vla.adapter import (  # noqa: E402
    _load_python_module,
    count_policy_parameters,
    retry_model_query,
)
from validator.modules.robotics_vla.errors import RoboticsSubmissionError  # noqa: E402
from validator.modules.robotics_vla.isolation import (  # noqa: E402
    decode_value,
    read_worker_message,
    write_worker_message,
)


def main() -> None:
    args = _parse_args()
    protocol_in = os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0)
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    os.close(devnull)

    try:
        _apply_resource_limits(
            args.memory_limit_bytes, args.cpu_time_seconds, args.device
        )
        _install_linux_filesystem_sandbox(Path(args.model_dir))
        _install_linux_seccomp()
        _apply_gpu_memory_limit(args.device, args.memory_limit_bytes)
        policy = _load_policy(args)
        parameter_count = _count_policy_parameters_safe(policy, args)
        write_worker_message(
            protocol_out,
            {"ok": True, "parameter_count": parameter_count},
        )
    except Exception as exc:  # noqa: BLE001 - report worker startup failures
        _write_error(protocol_out, exc, "model_load_failed")
        return

    while True:
        try:
            message = read_worker_message(protocol_in)
            operation = message.get("op")
            if operation == "close":
                return
            if operation != "act":
                raise RoboticsSubmissionError(
                    f"Unsupported policy operation {operation!r}",
                    failure_mode="policy_protocol_error",
                )
            obs = decode_value(message.get("obs"))
            action = policy.act(obs)
            array = np.asarray(action)
            write_worker_message(protocol_out, {"ok": True, "action": array.tolist()})
        except EOFError:
            return
        except Exception as exc:  # noqa: BLE001 - miner policy failures are isolated
            _write_error(protocol_out, exc, "policy_execution_failed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--adapter-filename", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--torch-dtype", required=True)
    parser.add_argument("--memory-limit-bytes", type=int, required=True)
    parser.add_argument("--cpu-time-seconds", type=int, required=True)
    return parser.parse_args()


def _load_policy(args: argparse.Namespace) -> Any:
    model_root = Path(args.model_dir).resolve()
    adapter_path = model_root / args.adapter_filename
    try:
        module = _load_python_module(adapter_path)
    except RoboticsSubmissionError:
        raise
    except Exception as exc:  # noqa: BLE001 - adapter is untrusted miner code
        raise RoboticsSubmissionError(
            f"Robotics VLA adapter failed to import: {exc}",
            failure_mode="adapter_import_failed",
        ) from exc
    if not hasattr(module, "load_policy"):
        raise RoboticsSubmissionError(
            f"{adapter_path} must define load_policy(model_dir, device, dtype)",
            failure_mode="adapter_contract",
        )

    policy = retry_model_query(
        lambda: module.load_policy(
            model_dir=str(model_root),
            device=args.device,
            dtype=args.torch_dtype,
        ),
        description="load_policy",
        failure_mode="model_load_failed",
    )
    if not hasattr(policy, "act"):
        raise RoboticsSubmissionError(
            "Robotics VLA policy must expose act(obs) -> action",
            failure_mode="adapter_contract",
        )
    return policy


def _count_policy_parameters_safe(
    policy: Any, args: argparse.Namespace
) -> int | None:
    """Best-effort parameter count for telemetry only.

    Model size is enforced at runtime by the host memory monitor and the CUDA
    allocator cap, not by this number, so it never raises and never gates: a
    submission that fits in the memory budget is allowed however many parameters
    it holds, and an unaccountable graph is reported as unknown rather than
    rejected.
    """
    try:
        return count_policy_parameters(
            policy,
            module_roots=(Path(args.model_dir).resolve(),),
        )
    except Exception:  # noqa: BLE001 - telemetry must not break action serving
        return None


def _apply_gpu_memory_limit(device: str, memory_limit_bytes: int) -> None:
    """Cap this process's CUDA allocator so an oversize model fails at load.

    Best-effort and a no-op off CUDA or without torch. The driver-enforced cap is
    the primary VRAM bound; the host-side memory monitor bounds system RAM. Raw,
    non-torch CUDA allocations are not covered by this cap (documented residual).
    """
    if not device.lower().startswith("cuda"):
        return
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    try:
        index = torch.device(device).index
    except Exception:  # noqa: BLE001 - fall back to the active device
        index = None
    if index is None:
        index = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(index).total_memory
    if total <= 0:
        return
    fraction = min(1.0, memory_limit_bytes / total)
    torch.cuda.set_per_process_memory_fraction(fraction, index)


def _write_error(stream: Any, exc: Exception, fallback_mode: str) -> None:
    write_worker_message(
        stream,
        {
            "ok": False,
            "error": str(exc),
            "failure_mode": getattr(exc, "failure_mode", fallback_mode),
        },
    )


def _apply_resource_limits(
    memory_limit_bytes: int, cpu_time_seconds: int, device: str
) -> None:
    limits = [
        (resource.RLIMIT_CPU, cpu_time_seconds),
        (resource.RLIMIT_FSIZE, 64 * 1024**2),
        (resource.RLIMIT_NOFILE, 64),
    ]
    # RLIMIT_AS bounds *virtual* address space. A CUDA context reserves far more
    # VA than it uses, so a model-sized RLIMIT_AS would break GPU init — there the
    # CUDA allocator cap bounds VRAM and the host memory monitor bounds system RAM.
    # On CPU the weights live in the address space, so keep the hard cap as a
    # kernel-enforced backstop. (Darwin refuses to lower an infinite RLIMIT_AS, so
    # it is Linux-only regardless.)
    if platform.system() == "Linux" and not device.lower().startswith("cuda"):
        limits.insert(0, (resource.RLIMIT_AS, memory_limit_bytes))
    for key, requested in limits:
        current_soft, current_hard = resource.getrlimit(key)
        hard = (
            requested
            if current_hard == resource.RLIM_INFINITY
            else min(requested, current_hard)
        )
        soft = min(requested, hard)
        # Darwin rejects lowering an infinite hard limit while the old soft
        # limit is still infinite. Lower the soft limit first, then lock hard.
        resource.setrlimit(key, (soft, current_hard))
        resource.setrlimit(key, (soft, hard))


# O_PATH descriptors kept open for the worker's lifetime; see
# _install_linux_filesystem_sandbox for why the /proc entries must stay pinned.
_PINNED_PROC_DESCRIPTORS: list[int] = []


def _install_linux_filesystem_sandbox(model_dir: Path) -> None:
    """Restrict miner file access with unprivileged Linux Landlock.

    The policy may read its model, Python/runtime libraries, device metadata, and
    its private temporary directory. It cannot read the validator checkout,
    parent process files, home credentials, or mutate submitted model files.
    """
    if platform.system() != "Linux":
        return

    LANDLOCK_CREATE_RULESET_VERSION = 1
    LANDLOCK_RULE_PATH_BENEATH = 1
    SYS_LANDLOCK_CREATE_RULESET = 444
    SYS_LANDLOCK_RESTRICT_SELF = 446
    PR_SET_NO_NEW_PRIVS = 38

    ACCESS_EXECUTE = 1 << 0
    ACCESS_WRITE_FILE = 1 << 1
    ACCESS_READ_FILE = 1 << 2
    ACCESS_READ_DIR = 1 << 3
    ACCESS_REMOVE_DIR = 1 << 4
    ACCESS_REMOVE_FILE = 1 << 5
    ACCESS_MAKE_CHAR = 1 << 6
    ACCESS_MAKE_DIR = 1 << 7
    ACCESS_MAKE_REG = 1 << 8
    ACCESS_MAKE_SOCK = 1 << 9
    ACCESS_MAKE_FIFO = 1 << 10
    ACCESS_MAKE_BLOCK = 1 << 11
    ACCESS_MAKE_SYM = 1 << 12
    ACCESS_REFER = 1 << 13
    ACCESS_TRUNCATE = 1 << 14
    ACCESS_IOCTL_DEV = 1 << 15

    class RulesetAttr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathBeneathAttr(ctypes.Structure):
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    abi = libc.syscall(
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.c_void_p(),
        0,
        LANDLOCK_CREATE_RULESET_VERSION,
    )
    if abi < 1:
        raise OSError(
            ctypes.get_errno(), "Landlock is unavailable on this Linux kernel"
        )

    handled = (
        ACCESS_EXECUTE
        | ACCESS_WRITE_FILE
        | ACCESS_READ_FILE
        | ACCESS_READ_DIR
        | ACCESS_REMOVE_DIR
        | ACCESS_REMOVE_FILE
        | ACCESS_MAKE_CHAR
        | ACCESS_MAKE_DIR
        | ACCESS_MAKE_REG
        | ACCESS_MAKE_SOCK
        | ACCESS_MAKE_FIFO
        | ACCESS_MAKE_BLOCK
        | ACCESS_MAKE_SYM
    )
    if abi >= 2:
        handled |= ACCESS_REFER
    if abi >= 3:
        handled |= ACCESS_TRUNCATE
    if abi >= 5:
        handled |= ACCESS_IOCTL_DEV

    ruleset_attr = RulesetAttr(handled)
    ruleset_fd = libc.syscall(
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(ruleset_attr),
        ctypes.sizeof(ruleset_attr),
        0,
    )
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")

    read_access = ACCESS_EXECUTE | ACCESS_READ_FILE | ACCESS_READ_DIR
    device_access = ACCESS_READ_FILE | ACCESS_READ_DIR | ACCESS_WRITE_FILE
    if abi >= 5:
        device_access |= ACCESS_IOCTL_DEV

    model_root = model_dir.resolve()
    read_roots = {
        model_root,
        Path(sys.executable).resolve().parent,
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
    }
    if model_root.parent.name == "snapshots":
        read_roots.add(model_root.parent.parent)
    for entry in sys.path:
        if not entry:
            continue
        candidate = Path(entry).expanduser().resolve()
        if candidate != _REPO_ROOT and _REPO_ROOT not in candidate.parents:
            read_roots.add(candidate)
    for variable in ("LD_LIBRARY_PATH",):
        for entry in os.getenv(variable, "").split(os.pathsep):
            if entry:
                read_roots.add(Path(entry).expanduser().resolve())
    read_roots.update(Path(path) for path in ("/usr", "/lib", "/lib64", "/etc"))

    try:
        for path in sorted(read_roots, key=str):
            if path.exists():
                _add_landlock_path_rule(
                    libc,
                    ruleset_fd,
                    LANDLOCK_RULE_PATH_BENEATH,
                    PathBeneathAttr,
                    path,
                    read_access,
                )
        temp_root = Path(os.environ["TMPDIR"]).resolve()
        _add_landlock_path_rule(
            libc,
            ruleset_fd,
            LANDLOCK_RULE_PATH_BENEATH,
            PathBeneathAttr,
            temp_root,
            handled,
        )
        # procfs builds a fresh inode for /proc/<pid> whenever its dentry is
        # dropped, so a rule attached to it silently stops matching. Hold O_PATH
        # descriptors on the worker's OWN /proc/<pid> and /proc/<pid>/task for its
        # whole lifetime so those dentries, and the rules on them, stay put. The
        # CUDA driver reads /proc/self/fd and names its threads through
        # /proc/self/task/<tid>/comm; without both, CUDA init fails with error 304
        # (verified on an L40, driver 580). No other process's /proc is granted.
        own_proc = Path(f"/proc/{os.getpid()}")
        for path, access in (
            (own_proc, read_access),
            (
                own_proc / "task",
                read_access | ACCESS_WRITE_FILE | (ACCESS_TRUNCATE & handled),
            ),
        ):
            descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
            _PINNED_PROC_DESCRIPTORS.append(descriptor)
            attr = PathBeneathAttr(access, descriptor)
            if (
                libc.syscall(
                    445,  # landlock_add_rule
                    ruleset_fd,
                    LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.byref(attr),
                    0,
                )
                != 0
            ):
                raise OSError(
                    ctypes.get_errno(), f"landlock_add_rule failed for {path}"
                )
        for path in (
            Path("/dev"),
            Path("/proc/driver/nvidia"),
            Path("/proc/cpuinfo"),
            Path("/proc/meminfo"),
            Path("/proc/stat"),
            # Read by the CUDA driver during initialisation; without it CUDA
            # init fails with error 304 (verified on an RTX A6000, driver 570).
            Path("/proc/sys/vm/mmap_min_addr"),
            Path("/sys"),
        ):
            if path.exists():
                access = device_access if path == Path("/dev") else read_access
                _add_landlock_path_rule(
                    libc,
                    ruleset_fd,
                    LANDLOCK_RULE_PATH_BENEATH,
                    PathBeneathAttr,
                    path,
                    access,
                )
        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
        if libc.syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset_fd)


def _add_landlock_path_rule(
    libc: Any,
    ruleset_fd: int,
    rule_type: int,
    attr_type: Any,
    path: Path,
    allowed_access: int,
) -> None:
    if path.is_file():
        # Directory-only and create/remove rights are invalid on file rules.
        allowed_access &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14) | (1 << 15)
    descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        attr = attr_type(allowed_access, descriptor)
        if (
            libc.syscall(
                445,
                ruleset_fd,
                rule_type,
                ctypes.byref(attr),
                0,
            )
            != 0
        ):
            raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
    finally:
        os.close(descriptor)


def _install_linux_seccomp() -> None:
    if platform.system() != "Linux":
        return

    architecture = platform.machine().lower()
    blocked = {
        "x86_64": {
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            57,
            58,
            59,
            101,
            272,
            288,
            299,
            307,
            308,
            310,
            311,
            322,
            425,
            426,
            427,
            438,
        },
        "amd64": {
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            57,
            58,
            59,
            101,
            272,
            288,
            299,
            307,
            308,
            310,
            311,
            322,
            425,
            426,
            427,
            438,
        },
        "aarch64": {
            97,
            117,
            198,
            199,
            200,
            201,
            202,
            203,
            204,
            205,
            206,
            207,
            208,
            209,
            210,
            211,
            212,
            221,
            242,
            243,
            268,
            269,
            270,
            271,
            281,
            425,
            426,
            427,
            438,
        },
        "arm64": {
            97,
            117,
            198,
            199,
            200,
            201,
            202,
            203,
            204,
            205,
            206,
            207,
            208,
            209,
            210,
            211,
            212,
            221,
            242,
            243,
            268,
            269,
            270,
            271,
            281,
            425,
            426,
            427,
            438,
        },
    }.get(architecture)
    if blocked is None:
        raise RuntimeError(f"Unsupported seccomp architecture: {architecture}")
    if architecture in {"x86_64", "amd64"}:
        blocked.add(56 | 0x40000000)  # deny the x32 clone ABI entirely
    # (socket, connect, bind, setsockopt, getsockopt, listen). The CUDA driver
    # needs a local AF_UNIX socket during initialisation: it probes for an MPS
    # control daemon (connect, ENOENT when absent) and binds and listens on an
    # abstract "cuda-uvmfd-*" socket. Denying these makes CUDA init fail with
    # error 304. socket/connect get dedicated rules below; bind/setsockopt/
    # getsockopt/listen are allowed because they only ever act on an AF_UNIX
    # socket (every other address family is refused at socket()), and accept
    # stays blocked so a listening socket can never take a connection. Their
    # x32 aliases stay blocked.
    unix_socket_calls = (
        (41, 42, 49, 54, 55, 50)
        if architecture in {"x86_64", "amd64"}
        else (198, 203, 200, 208, 209, 201)
    )
    socket_nr, connect_nr = unix_socket_calls[0], unix_socket_calls[1]
    blocked.difference_update(unix_socket_calls)
    if architecture in {"x86_64", "amd64"}:
        blocked.update(nr | 0x40000000 for nr in unix_socket_calls)
    AF_UNIX = 1

    # Classic BPF: load syscall number, return EPERM for each blocked syscall,
    # and allow everything else. CUDA ioctls and normal threading remain usable.
    BPF_LD = 0x00
    BPF_W = 0x00
    BPF_ABS = 0x20
    BPF_JMP = 0x05
    BPF_JEQ = 0x10
    BPF_ALU = 0x04
    BPF_AND = 0x50
    BPF_K = 0x00
    BPF_RET = 0x06
    SECCOMP_RET_ALLOW = 0x7FFF0000
    SECCOMP_RET_ERRNO = 0x00050000
    PR_SET_NO_NEW_PRIVS = 38
    PR_SET_SECCOMP = 22
    SECCOMP_MODE_FILTER = 2

    class SockFilter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint32),
        ]

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]

    clone_syscall = 56 if architecture in {"x86_64", "amd64"} else 220
    audit_arch = 0xC000003E if architecture in {"x86_64", "amd64"} else 0xC00000B7
    CLONE_THREAD = 0x00010000
    # Only thread creation is permitted. A child process would otherwise get a
    # fresh per-process CPU/address-space budget and could be used as a fork bomb.
    instructions = [
        SockFilter(BPF_LD | BPF_W | BPF_ABS, 0, 0, 4),  # seccomp_data.arch
        SockFilter(BPF_JMP | BPF_JEQ | BPF_K, 1, 0, audit_arch),
        SockFilter(BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ERRNO | errno.EPERM),
        SockFilter(BPF_LD | BPF_W | BPF_ABS, 0, 0, 0),
        SockFilter(BPF_JMP | BPF_JEQ | BPF_K, 0, 4, clone_syscall),
        SockFilter(BPF_LD | BPF_W | BPF_ABS, 0, 0, 16),  # args[0], low 32 bits
        SockFilter(BPF_ALU | BPF_AND | BPF_K, 0, 0, CLONE_THREAD),
        SockFilter(BPF_JMP | BPF_JEQ | BPF_K, 1, 0, CLONE_THREAD),
        SockFilter(BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ERRNO | errno.EPERM),
        SockFilter(BPF_LD | BPF_W | BPF_ABS, 0, 0, 0),
        # Make glibc fall back from clone3 to the flag-checked clone syscall.
        SockFilter(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, 435),
        SockFilter(BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ERRNO | errno.ENOSYS),
        # socket(): AF_UNIX only, so no IP/netlink/packet socket can ever exist.
        SockFilter(BPF_JMP | BPF_JEQ | BPF_K, 0, 4, socket_nr),
        SockFilter(BPF_LD | BPF_W | BPF_ABS, 0, 0, 16),  # args[0] = domain
        SockFilter(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, AF_UNIX),
        SockFilter(BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ALLOW),
        SockFilter(BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ERRNO | errno.EPERM),
        # connect(): always "no such socket". Landlock does not mediate connecting
        # to pathname UNIX sockets, so allowing connect would expose host daemons
        # (e.g. docker.sock). ENOENT is exactly what CUDA sees when no MPS daemon
        # runs, so GPU init proceeds normally. sendto/sendmsg stay blocked, so an
        # unconnected datagram socket cannot reach anything either.
        SockFilter(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, connect_nr),
        SockFilter(BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ERRNO | errno.ENOENT),
    ]
    for syscall_number in sorted(blocked):
        instructions.append(SockFilter(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, syscall_number))
        instructions.append(
            SockFilter(BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ERRNO | errno.EPERM)
        )
        if architecture in {"x86_64", "amd64"}:
            instructions.append(
                SockFilter(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, syscall_number | 0x40000000)
            )
            instructions.append(
                SockFilter(BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ERRNO | errno.EPERM)
            )
    instructions.append(SockFilter(BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ALLOW))
    program_array = (SockFilter * len(instructions))(*instructions)
    program = SockFprog(len(instructions), program_array)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(program)) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_SECCOMP) failed")


if __name__ == "__main__":
    main()
