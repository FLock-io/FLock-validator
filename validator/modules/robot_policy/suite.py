"""Download and validate the server-selected RobotPolicy evaluation suite."""
from __future__ import annotations

import ipaddress
import math
import os
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import urlsplit

import requests
import yaml
from yaml.events import AliasEvent
from yaml.nodes import MappingNode, ScalarNode


MAX_SUITE_BYTES = 1024 * 1024
DOWNLOAD_DEADLINE_S = 30.0
REQUEST_TIMEOUT = (5, 10)
_PHASES = frozenset({"training", "final_validation"})
_INITIAL_STATES = {
    "Mjlab-Velocity-Flat-MicroDuck": frozenset({"standing"}),
    "Mjlab-StandUp-Flat-MicroDuck": frozenset({"face_down", "face_up", "sitting", "standing"}),
}
_REQUIRED_FIELDS = frozenset({
    "suite_id", "suite_type", "episodes", "suite_seed", "timeout_s",
    "scenario_distribution", "mjlab_task_id",
})


class SuiteError(RuntimeError):
    """The server's suite is unavailable or violates the benchmark contract."""


class _SuiteLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        if self.check_event(AliasEvent):
            raise SuiteError("suite YAML aliases are not allowed")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, MappingNode):
            raise SuiteError("suite YAML must contain mappings")
        result = {}
        for key_node, value_node in node.value:
            if not isinstance(key_node, ScalarNode) or key_node.tag != "tag:yaml.org,2002:str":
                raise SuiteError("suite mapping keys must be strings")
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise SuiteError("suite YAML contains duplicate keys")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def validate_suite_url(url: str) -> str:
    """Validate a signed HTTPS URL without exposing it in error messages.

    DNS names are not resolved here. Literal addresses must be public, and
    alternate numeric spellings are refused rather than treated as DNS names.
    Query strings are preserved exactly because they can carry signatures.
    """
    invalid = "suite URL must use HTTPS with a public host and no credentials or fragment"
    if (not isinstance(url, str) or not url or url != url.strip()
            or "#" in url or "\\" in url
            or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in url)):
        raise SuiteError(invalid)
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
        if (parsed.scheme.lower() != "https" or not hostname
                or parsed.username is not None or parsed.password is not None
                or (port is not None and not 1 <= port <= 65535)):
            raise ValueError
        host = hostname.rstrip(".").lower()
        if "%" in host:
            raise ValueError
        # Match the normalization performed by HTTP clients before applying
        # address restrictions (full-width localhost/digits must not bypass it).
        host = host.encode("idna").decode("ascii").lower().rstrip(".")
        if not host or host == "localhost" or host.endswith(".localhost"):
            raise ValueError
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            # libc and HTTP stacks can interpret these as noncanonical IPs.
            if all(re.fullmatch(r"(?:0x[0-9a-f]+|[0-9]+)", part) for part in host.split(".")):
                raise ValueError
            if len(host) > 253 or any(
                not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
                for label in host.split(".")
            ):
                raise ValueError
        else:
            if (not address.is_global or address.is_multicast or address.is_reserved
                    or address.is_loopback or address.is_link_local or address.is_unspecified):
                raise ValueError
    except (ValueError, UnicodeError):
        raise SuiteError(invalid) from None
    return url


def _context(task_id, validation_phase):
    if not isinstance(task_id, str) or task_id not in _INITIAL_STATES:
        raise SuiteError("unsupported RobotPolicy task configuration")
    if validation_phase is not None and (
            not isinstance(validation_phase, str) or validation_phase not in _PHASES):
        raise SuiteError("unsupported validation phase")


def _mapping(value, required, optional, label):
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - optional:
        raise SuiteError(f"{label} has missing or unsupported fields")


def _text(value, label):
    if (not isinstance(value, str) or len(value) > 128
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)):
        raise SuiteError(f"{label} must be a safe identifier of 1 to 128 characters")


def _integer(value, minimum, maximum, label):
    if type(value) is not int or not minimum <= value <= maximum:
        raise SuiteError(f"{label} must be an integer between {minimum} and {maximum}")


def _positive(value, maximum, label):
    if type(value) not in (int, float):
        raise SuiteError(f"{label} must be a positive finite number no greater than {maximum}")
    try:
        valid = math.isfinite(value) and 0 < value <= maximum
    except OverflowError:
        valid = False
    if not valid:
        raise SuiteError(f"{label} must be a positive finite number no greater than {maximum}")


def _perturbation(value, name):
    if name == "command_delay_steps":
        if isinstance(value, list):
            if not value:
                raise SuiteError("command_delay_steps list must not be empty")
            for item in value:
                _integer(item, 0, 3, name)
        else:
            _integer(value, 0, 3, name)
        return
    maximum = 5 if name == "friction" else 20
    if isinstance(value, dict):
        _mapping(value, {"min", "max"}, set(), name)
        _positive(value["min"], maximum, name)
        _positive(value["max"], maximum, name)
        if value["min"] > value["max"]:
            raise SuiteError(f"{name} range must have min <= max")
    elif isinstance(value, list):
        if not value:
            raise SuiteError(f"{name} list must not be empty")
        for item in value:
            _positive(item, maximum, name)
    else:
        _positive(value, maximum, name)


def parse_suite(content: bytes, *, task_id: str, validation_phase: str | None = None) -> dict:
    """Parse UTF-8 YAML and reject any unsupported benchmark semantics."""
    _context(task_id, validation_phase)
    if not isinstance(content, bytes) or not content or len(content) > MAX_SUITE_BYTES:
        raise SuiteError("suite content must be nonempty and no larger than 1 MiB")
    try:
        suite = yaml.load(content.decode("utf-8"), Loader=_SuiteLoader)
    except SuiteError:
        raise
    except (UnicodeError, yaml.YAMLError, ValueError, TypeError, RecursionError):
        raise SuiteError("suite content must be valid UTF-8 YAML") from None
    _mapping(suite, _REQUIRED_FIELDS, {"suite_family", "validation_phase"}, "suite")
    _text(suite["suite_id"], "suite_id")
    if "suite_family" in suite:
        _text(suite["suite_family"], "suite_family")
    if suite["suite_type"] not in ("standard", "robustness"):
        raise SuiteError("suite_type must be standard or robustness")
    if suite["mjlab_task_id"] != task_id:
        raise SuiteError("suite task does not match the assigned RobotPolicy task")
    if "validation_phase" in suite and (
            not isinstance(suite["validation_phase"], str) or suite["validation_phase"] not in _PHASES):
        raise SuiteError("unsupported suite validation phase")
    if validation_phase is not None and suite.get("validation_phase") != validation_phase:
        raise SuiteError("suite phase does not match the requested phase")
    _integer(suite["episodes"], 1, 10000, "episodes")
    _integer(suite["suite_seed"], 0, 2**64 - 1, "suite_seed")
    _positive(suite["timeout_s"], 300, "timeout_s")
    steps = suite["timeout_s"] * 50
    if round(steps) < 1 or not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-8):
        raise SuiteError("timeout_s must contain a whole number of 50 Hz control steps")
    distribution = suite["scenario_distribution"]
    _mapping(distribution, {"initial_states"}, {"perturbations"}, "scenario_distribution")
    states = distribution["initial_states"]
    if (not isinstance(states, list) or not states
            or any(not isinstance(state, str) or state not in _INITIAL_STATES[task_id] for state in states)):
        raise SuiteError("initial_states must be a nonempty list supported by the assigned task")
    perturbations = distribution.get("perturbations", {})
    _mapping(perturbations, set(), {"friction", "command_delay_steps", "battery_voltage"}, "perturbations")
    for name, value in perturbations.items():
        _perturbation(value, name)
    return suite


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise SuiteError("suite download exceeded its total deadline")


def fetch_suite(url: str, destination: Path, *, task_id: str, validation_phase: str | None = None) -> dict:
    """Download one server suite; persist only successfully validated bytes."""
    _context(task_id, validation_phase)
    url = validate_suite_url(url)
    deadline = time.monotonic() + DOWNLOAD_DEADLINE_S
    content = bytearray()
    try:
        with requests.get(url, stream=True, allow_redirects=False, timeout=REQUEST_TIMEOUT) as response:
            _check_deadline(deadline)
            if response.status_code != 200:
                raise SuiteError("suite download requires HTTP 200; redirects are not accepted")
            length = response.headers.get("Content-Length")
            expected = None
            if length is not None:
                try:
                    expected = int(length)
                except (ValueError, TypeError):
                    raise SuiteError("suite download has an invalid content length") from None
                if expected < 0 or expected > MAX_SUITE_BYTES:
                    raise SuiteError("suite download exceeds the 1 MiB limit")
            # Small reads keep deadline checks active even for a trickled suite.
            # The socket read timeout additionally bounds a stalled connection.
            for chunk in response.iter_content(chunk_size=1):
                _check_deadline(deadline)
                if len(content) + len(chunk) > MAX_SUITE_BYTES:
                    raise SuiteError("suite download exceeds the 1 MiB limit")
                content.extend(chunk)
            _check_deadline(deadline)
            encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
            if expected is not None and encoding in ("", "identity") and len(content) != expected:
                raise SuiteError("suite download content length does not match the response")
    except SuiteError:
        raise
    except requests.RequestException:
        # Requests exceptions can include a signed URL. Never chain or echo it.
        raise SuiteError("suite download failed or timed out") from None
    raw = bytes(content)
    suite = parse_suite(raw, task_id=task_id, validation_phase=validation_phase)
    destination = Path(destination)
    temporary = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".robot_policy-suite-",
                                         suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(raw)
        os.replace(temporary, destination)
    except OSError:
        raise SuiteError("validated suite could not be saved") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # Preserve the download/save exception if cleanup also fails.
                pass
    return suite
