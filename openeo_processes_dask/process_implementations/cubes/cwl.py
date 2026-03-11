import json
import logging
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from openeo_processes_dask.process_implementations.exceptions import (
    CwlExecutionFailed,
    CwlInputMismatch,
    CwlInvalidDocument,
)

__all__ = ["run_cwl"]

logger = logging.getLogger(__name__)


def _is_url(value: str) -> bool:
    """Check if a string looks like a URL."""
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https")


def _resolve_cwl(cwl: str, work_dir: Path) -> Path:
    """Resolve a CWL argument to a local file path.

    If cwl is a URL, download it. If it's an inline string, write it to a
    temporary file. Returns the path to the CWL file on disk.
    """
    if _is_url(cwl):
        import urllib.request

        cwl_path = work_dir / "workflow.cwl"
        urllib.request.urlretrieve(cwl, str(cwl_path))
        logger.info(f"Downloaded CWL from {cwl} to {cwl_path}")
        return cwl_path

    # Inline CWL string — write to file
    cwl_path = work_dir / "workflow.cwl"
    cwl_path.write_text(cwl)
    logger.info(f"Wrote inline CWL to {cwl_path}")
    return cwl_path


def _write_inputs(inputs: dict, work_dir: Path) -> Path:
    """Write CWL job inputs to a JSON file."""
    inputs_path = work_dir / "inputs.json"
    inputs_path.write_text(json.dumps(inputs, indent=2))
    return inputs_path


def _validate_cwl(cwl_path: Path) -> dict:
    """Validate a CWL document using cwltool.

    Returns a dict with 'valid' (bool) and 'errors' (list of str).
    """
    import subprocess

    result = subprocess.run(
        ["cwltool", "--validate", str(cwl_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode == 0:
        return {"valid": True, "errors": []}

    errors = []
    for line in (result.stderr or "").strip().splitlines():
        if line.strip():
            errors.append(line.strip())

    return {"valid": False, "errors": errors}


def run_cwl(
    cwl: str,
    inputs: dict,
    options: Optional[dict] = None,
):
    """Execute a CWL workflow or CommandLineTool.

    This is a stub implementation for the process contract. The actual
    execution through Calrissian is handled by the executor backend
    (openeo-argoworkflows), which intercepts this process in the
    process graph and routes it to the CWL execution path.

    This function handles:
    - CWL resolution (inline string or URL)
    - CWL validation via cwltool
    - Input file preparation
    - validate_only mode (returns validation result without execution)

    The executor backend is responsible for:
    - Submitting to Calrissian on Kubernetes
    - Collecting outputs
    - Mapping outputs to openEO job results

    Parameters
    ----------
    cwl : str
        Inline CWL document (YAML/JSON) or URL to a remote CWL file.
    inputs : dict
        Key-value mapping of CWL input parameter names to values.
    options : dict, optional
        Execution options. Supported keys:
        - validate_only (bool): Validate without executing. Default False.
        - max_ram (str): Max RAM for CWL steps, e.g. "8G".
        - max_cores (int): Max CPU cores for CWL steps.

    Returns
    -------
    dict
        If validate_only: {"valid": bool, "errors": list[str]}
        Otherwise: result is handled by the executor backend.

    Raises
    ------
    CwlInvalidDocument
        If the CWL document fails validation.
    CwlInputMismatch
        If the provided inputs don't match the CWL's declared inputs.
    CwlExecutionFailed
        If the CWL execution fails.
    """
    if options is None:
        options = {}

    validate_only = options.get("validate_only", False)

    with tempfile.TemporaryDirectory(prefix="openeo_cwl_") as work_dir:
        work_dir = Path(work_dir)

        # Resolve CWL to a local file
        try:
            cwl_path = _resolve_cwl(cwl, work_dir)
        except Exception as e:
            raise CwlInvalidDocument(f"Failed to resolve CWL document: {e}")

        # Validate CWL
        validation = _validate_cwl(cwl_path)

        if not validation["valid"]:
            raise CwlInvalidDocument(
                f"CWL validation failed: {'; '.join(validation['errors'])}"
            )

        if validate_only:
            return validation

        # Write inputs file
        inputs_path = _write_inputs(inputs, work_dir)

        # Actual execution is handled by the executor backend.
        # When the executor encounters run_cwl in the process graph,
        # it calls the CWL execution path (Calrissian) directly,
        # bypassing this function's return value.
        #
        # If we reach here in a non-executor context (e.g. local testing),
        # we raise an error indicating that CWL execution requires the
        # backend executor.
        raise CwlExecutionFailed(
            "CWL execution requires the openeo-argoworkflows executor backend "
            "with Calrissian configured. This process cannot run in a standalone "
            "openeo-processes-dask context. Use validate_only=true to validate "
            "CWL documents without execution."
        )
