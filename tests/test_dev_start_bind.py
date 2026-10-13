"""scripts/dev-start.sh binds loopback unless the caller opts in.

Dev mode usually runs with ``CF_ACCESS_DISABLED=1`` (no auth), so a
default bind on every interface hands an open API to the LAN. The
script runs here with stub ``uv`` / ``npm`` / ``tailscale`` binaries that
record the host they were told to bind.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "dev-start.sh"


def _run_script(tmp_path: Path, **env: str) -> tuple[str, str]:
    """Run dev-start.sh with stubbed tools; return (recorded calls, stdout)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    stubs = {
        "uv": 'echo "uv host=$AIDM_SERVER_HOST args=$*" >> "$STUB_LOG"',
        "npm": 'echo "npm args=$*" >> "$STUB_LOG"',
        "tailscale": "echo 203.0.113.10",
    }
    for name, body in stubs.items():
        stub = bin_dir / name
        stub.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        stub.chmod(0o755)
    full_env = {k: v for k, v in os.environ.items() if k != "AIDM_SERVER_HOST"}
    full_env.update(env)
    full_env["PATH"] = f"{bin_dir}{os.pathsep}{full_env['PATH']}"
    full_env["STUB_LOG"] = str(log)
    done = subprocess.run(
        ["bash", str(SCRIPT), "public", "webrtc"],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return log.read_text(encoding="utf-8"), done.stdout


def test_dev_start_binds_loopback_by_default(tmp_path: Path) -> None:
    calls, out = _run_script(tmp_path)
    assert "uv host=127.0.0.1 " in calls
    assert "--host 127.0.0.1" in calls
    assert "0.0.0.0" not in calls
    assert "AIDM_SERVER_HOST=0.0.0.0" in out  # the one-line opt-in hint


def test_dev_start_exposes_the_lan_only_when_asked(tmp_path: Path) -> None:
    calls, out = _run_script(tmp_path, AIDM_SERVER_HOST="0.0.0.0")
    assert "uv host=0.0.0.0 " in calls
    assert "--host 0.0.0.0" in calls
    assert "http://203.0.113.10:5173" in out
    assert "no auth" in out.lower()
