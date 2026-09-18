"""Model provenance and compatibility gates for persisted calculation results.

Numerical arrays alone cannot identify their physics. All production readers
validate this envelope before accepting saved phases or spectra. Historical
unversioned files are deliberately readable only through explicitly low-level
inspection functions, never through a calculation entry point.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .config import SourceConfig
from .detector_response import IntegrationSettings

MODEL_VERSION = "conserved-quadrupole-free-michelson-v1"
PHASOR_CONVENTION = "peak; Re[H exp(-i Omega t)]; rotor offset is time delay"


def model_metadata(config=None, settings=None):
    """JSON-compatible physical and numerical identity of a calculation."""
    return {
        "model_version": MODEL_VERSION,
        "phasor_convention": PHASOR_CONVENTION,
        "source_config": asdict(config or SourceConfig()),
        "integration_settings": asdict(settings or IntegrationSettings()),
    }


def validate_metadata(metadata, config=None, settings=None):
    """Reject old physics, phases, or settings instead of silently reusing them."""
    if not isinstance(metadata, dict):
        raise ValueError("Invalid artifact metadata; regenerate the artifact")
    expected = model_metadata(config, settings)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Incompatible artifact ({key}); regenerate geometry, source array and spectra with the current configuration"
            )


def sidecar_path(path):
    """Keep metadata adjacent to its payload, e.g. array.csv.metadata.json."""
    return Path(str(path) + ".metadata.json")


def write_metadata(path, metadata):
    """Write metadata after the payload has been successfully produced."""
    target = sidecar_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def read_metadata(path):
    """Missing metadata identifies an obsolete or incomplete artifact."""
    try:
        metadata = json.loads(sidecar_path(path).read_text())
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        return metadata
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Missing/invalid metadata for {path}; regenerate this artifact"
        ) from exc


def file_digest(path):
    """Streaming digest binds paired spectrum files without loading them twice."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
