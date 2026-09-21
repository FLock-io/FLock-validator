"""ONNX artifact checks, independent of the physics stack."""
from __future__ import annotations

import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil

from .errors import InvalidSubmission, ModelParameterLimit


def repo_filename(filename: str) -> str:
    if not isinstance(filename, str) or not filename:
        raise InvalidSubmission("model/external data filename must be relative to the HF repo")
    path = PurePosixPath(filename)
    # HF uses POSIX names, but the same artifacts may be staged on Windows.
    # C:/weights and C:weights are relative to PurePosixPath while Windows
    # interprets them as drive paths; ':' can also select an NTFS data stream.
    windows = PureWindowsPath(filename)
    if (not path.parts or path.is_absolute() or windows.drive or windows.root
            or ".." in path.parts or "\\" in filename or ":" in filename
            or any(ord(char) < 32 for char in filename)
            or any(part.endswith((".", " ")) for part in path.parts)):
        raise InvalidSubmission("model/external data filename must be relative to the HF repo")
    return str(path)


def _tensors(graph):
    yield from graph.initializer
    for sparse in graph.sparse_initializer:
        yield sparse.values
        yield sparse.indices
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                yield attr.t
            yield from attr.tensors
            if attr.HasField("g"):
                yield from _tensors(attr.g)
            for child in attr.graphs:
                yield from _tensors(child)


def load_policy_metadata(path):
    import onnx
    from google.protobuf.message import DecodeError
    try:
        return onnx.load(str(path), load_external_data=False)
    except DecodeError as exc:
        raise InvalidSubmission("not a valid ONNX protobuf") from exc


def external_files(model, filename):
    """Resolve only declared external tensors, including models in subfolders."""
    files = set()
    for tensor in _tensors(model.graph):
        if tensor.data_location == tensor.EXTERNAL:
            fields = {item.key: item.value for item in tensor.external_data}
            location = repo_filename(fields.get("location", ""))
            files.add(str(PurePosixPath(filename).parent / location))
    return sorted(files)


def check_policy(path, max_params=None):
    import onnx
    model = load_policy_metadata(path)
    # Match the reference RL module: number of elements in graph initializers.
    count = sum(math.prod(tensor.dims) for tensor in model.graph.initializer)
    if max_params is not None and count > max_params:
        raise ModelParameterLimit(f"Model parameters {count} exceed limit {max_params}")
    try:
        onnx.checker.check_model(str(path))
    except onnx.checker.ValidationError as exc:
        raise InvalidSubmission(f"invalid ONNX graph: {exc}") from exc
    return count


def stage_policy(path, destination):
    """Materialize HF cache links: ONNX checkers can reject symlinked tensors."""
    path, destination = Path(path), Path(destination)
    files = [path.name, *external_files(load_policy_metadata(path), path.name)]
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path.parent / name, target)
    return destination / path.name


def check_session_contract(session):
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise InvalidSubmission("off-contract: expected one observation input and one action output")
    for node, width, label in ((inputs[0], 61, "observation"), (outputs[0], 14, "action")):
        shape = node.shape
        if (len(shape) != 2 or shape[1] != width or node.type != "tensor(float)"
                or (isinstance(shape[0], int) and shape[0] != 1)):
            raise InvalidSubmission(
                f"off-contract: {label} must be float32 [1 or dynamic batch, {width}], "
                f"got {node.type} {shape}")
