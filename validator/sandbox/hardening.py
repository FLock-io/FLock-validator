"""Worker-side and host-side hardening primitives for sandboxed workers.

The ``install_*`` / ``apply_*`` functions run INSIDE the worker process, before
any untrusted code is imported. ``protect_parent_secrets`` and
``sandbox_environment`` run in the trusted parent. Everything here is generic:
task-specific workers decide which directories to grant.

Failures raise ``OSError``/``RuntimeError``; callers report them as "sandbox
unavailable" (an infrastructure problem), never as a submission fault.
"""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import resource
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from validator.sandbox.errors import SandboxUnavailableError

# validator/sandbox/hardening.py -> repository root. Landlock must never grant the
# validator checkout to a worker: it holds the validator source and, in a typical
# deployment, credentials and configuration.
_REPO_ROOT = Path(__file__).resolve().parents[2]

_SANDBOX_ENV_ALLOWLIST = {
    "CUDA_VISIBLE_DEVICES",
    "DYLD_LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "MKL_NUM_THREADS",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "PATH",
    "TOKENIZERS_PARALLELISM",
}


def sandbox_environment(temp_dir: Path) -> dict[str, str]:
    """Clean environment for a worker: an allowlist, never the parent's secrets.

    The worker has no network, so Hugging Face libraries are forced offline: a
    model that tries to download at load time then fails immediately and clearly
    instead of hanging on connection retries until the load timeout.
    """
    env = {
        key: value for key, value in os.environ.items() if key in _SANDBOX_ENV_ALLOWLIST
    }
    env.update(
        {
            "HOME": str(temp_dir),
            "TMPDIR": str(temp_dir),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    return env


def protect_parent_secrets() -> None:
    """Prevent same-UID workers from reading the parent through /proc on Linux."""
    if platform.system() != "Linux":
        return
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_DUMPABLE) failed")
    except (AttributeError, OSError) as exc:
        raise SandboxUnavailableError(
            f"Could not protect validator credentials from the sandbox worker: {exc}"
        ) from exc


# Per-file size cap for anything the worker writes (its TMPDIR/HOME only). Large
# enough for CUDA JIT, Triton and torch.compile caches or an unpacked weights
# shard, which a smaller cap would kill with SIGXFSZ mid-load.
_MAX_FILE_BYTES = 4 * 1024**3
# Sharded weights, CUDA device handles, video decoders and the pinned /proc
# descriptors all count against this; 64 broke large models with EMFILE.
_MAX_OPEN_FILES = 1024


def apply_resource_limits(
    memory_limit_bytes: int, cpu_time_seconds: int, device: str
) -> None:
    limits = [
        (resource.RLIMIT_CPU, cpu_time_seconds),
        (resource.RLIMIT_FSIZE, _MAX_FILE_BYTES),
        (resource.RLIMIT_NOFILE, _MAX_OPEN_FILES),
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


def apply_cuda_memory_limit(device: str, memory_limit_bytes: int) -> None:
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


# O_PATH descriptors kept open for the worker's lifetime; see install_landlock
# for why the worker's own /proc entries must stay pinned.
_PINNED_PROC_DESCRIPTORS: list[int] = []


def install_landlock(
    model_dir: Path,
    *,
    temp_dir: Path,
    extra_read_only_dirs: Iterable[Path] = (),
) -> None:
    """Restrict worker file access with unprivileged Linux Landlock.

    The worker may read its model, Python/runtime libraries, device metadata and
    any ``extra_read_only_dirs``, and read/write its private ``temp_dir``. It
    cannot read the validator checkout, parent process files or home credentials,
    nor mutate the submitted model files or the read-only input directories.
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
        # Interpreter and site-packages paths are needed; the validator checkout
        # (the repo root and everything below it) is deliberately excluded.
        if candidate != _REPO_ROOT and _REPO_ROOT not in candidate.parents:
            read_roots.add(candidate)
    for variable in ("LD_LIBRARY_PATH",):
        for entry in os.getenv(variable, "").split(os.pathsep):
            if entry:
                read_roots.add(Path(entry).expanduser().resolve())
    read_roots.update(Path(path) for path in ("/usr", "/lib", "/lib64", "/etc"))

    extra_roots: list[Path] = []
    for extra in extra_read_only_dirs:
        extra_root = Path(extra).resolve()
        if not extra_root.exists():
            # A read-only grant that silently vanished would surface later as a
            # confusing EACCES inside untrusted code; fail the hardening instead.
            raise FileNotFoundError(f"Read-only sandbox path missing: {extra_root}")
        extra_roots.append(extra_root)

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
        for path in extra_roots:
            _add_landlock_path_rule(
                libc,
                ruleset_fd,
                LANDLOCK_RULE_PATH_BENEATH,
                PathBeneathAttr,
                path,
                read_access,
            )
        _add_landlock_path_rule(
            libc,
            ruleset_fd,
            LANDLOCK_RULE_PATH_BENEATH,
            PathBeneathAttr,
            Path(temp_dir).resolve(),
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
                raise OSError(ctypes.get_errno(), f"landlock_add_rule failed for {path}")
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
                445,  # landlock_add_rule
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


# Syscalls answered with EPERM. Sockets (no network; socket/connect/bind and
# get/setsockopt are then re-handled in install_seccomp so the CUDA driver can
# create a local AF_UNIX socket that can never reach anything), exec/fork (no child
# processes), ptrace / process_vm_* (no poking other processes), unshare / setns
# (no namespace games), io_uring (bypasses per-syscall filtering) and pidfd_getfd.
_BLOCKED_X86_64: frozenset[int] = frozenset(
    {
        *range(41, 56),  # socket .. getsockopt (connect, accept, sendto, ...)
        57,  # fork
        58,  # vfork
        59,  # execve
        101,  # ptrace
        272,  # unshare
        288,  # accept4
        299,  # recvmmsg
        307,  # sendmmsg
        308,  # setns
        310,  # process_vm_readv
        311,  # process_vm_writev
        322,  # execveat
        425,  # io_uring_setup
        426,  # io_uring_enter
        427,  # io_uring_register
        438,  # pidfd_getfd
    }
)
_BLOCKED_AARCH64: frozenset[int] = frozenset(
    {
        97,  # unshare
        117,  # ptrace
        *range(198, 213),  # socket .. recvmsg
        221,  # execve
        242,  # accept4
        243,  # recvmmsg
        268,  # setns
        269,  # sendmmsg
        270,  # process_vm_readv
        271,  # process_vm_writev
        281,  # execveat
        425,  # io_uring_setup
        426,  # io_uring_enter
        427,  # io_uring_register
        438,  # pidfd_getfd
    }
)
# (socket, connect, bind, setsockopt, getsockopt, listen) per architecture. The
# CUDA driver needs a local AF_UNIX socket during initialisation: it probes for
# an MPS control daemon (connect to /tmp/nvidia-mps/control, ENOENT when absent)
# and binds and listens on an abstract "cuda-uvmfd-*" socket. Denying these makes
# cudaGetDeviceCount fail with error 304 (verified on an RTX A6000 with driver 570
# and an L40 with driver 580). accept stays blocked, so a listening socket can
# never take a connection.
_UNIX_SOCKET_SYSCALLS: Mapping[str, tuple[int, ...]] = {
    "x86_64": (41, 42, 49, 54, 55, 50),
    "amd64": (41, 42, 49, 54, 55, 50),
    "aarch64": (198, 203, 200, 208, 209, 201),
    "arm64": (198, 203, 200, 208, 209, 201),
}
_AF_UNIX = 1

_BLOCKED_SYSCALLS: Mapping[str, frozenset[int]] = {
    "x86_64": _BLOCKED_X86_64,
    "amd64": _BLOCKED_X86_64,
    "aarch64": _BLOCKED_AARCH64,
    "arm64": _BLOCKED_AARCH64,
}


def install_seccomp() -> None:
    """Install a seccomp-bpf filter: no network, no exec, no fork; threads only.

    Only local AF_UNIX sockets may be created (the CUDA driver needs one) and
    they can never connect or send, so the worker has no channel to anything.
    """
    if platform.system() != "Linux":
        return

    architecture = platform.machine().lower()
    base_blocked = _BLOCKED_SYSCALLS.get(architecture)
    if base_blocked is None:
        raise RuntimeError(f"Unsupported seccomp architecture: {architecture}")
    blocked = set(base_blocked)
    is_x86 = architecture in {"x86_64", "amd64"}
    if is_x86:
        blocked.add(56 | 0x40000000)  # deny the x32 clone ABI entirely
    unix_socket_calls = _UNIX_SOCKET_SYSCALLS[architecture]
    socket_nr, connect_nr = unix_socket_calls[0], unix_socket_calls[1]
    # socket/connect get dedicated rules below; bind/setsockopt/getsockopt/listen
    # are allowed because they are only reachable on an AF_UNIX socket (every
    # other address family is refused at socket()). Their x32 aliases stay blocked.
    blocked.difference_update(unix_socket_calls)
    if is_x86:
        blocked.update(nr | 0x40000000 for nr in unix_socket_calls)

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

    clone_syscall = 56 if is_x86 else 220
    audit_arch = 0xC000003E if is_x86 else 0xC00000B7
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
        SockFilter(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, _AF_UNIX),
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
        if is_x86:
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
