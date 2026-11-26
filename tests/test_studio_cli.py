"""Local studio + CLI smoke tests (no network)."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

from typer.testing import CliRunner

from trialgeist.bench.synthetic import make_trial
from trialgeist.cli import app


def test_server_scan_and_export(tmp_path: Path):
    from trialgeist.server import make_server

    t = make_trial(4242, tmp_path, "percent")
    srv = make_server("127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert b"TRIALGEIST studio" in urllib.request.urlopen(base + "/").read()
        req = urllib.request.Request(base + "/api/scan", data=t.pdf.read_bytes(), method="POST")
        rep = json.loads(urllib.request.urlopen(req).read())
        proven = [f for f in rep["findings"] if f["tier"] == "PROVEN_IMPOSSIBLE"]
        assert proven and all(f["provenance"][0]["bbox"] for f in proven)
        assert proven[0]["stats"]["certificate_seconds"] < 2
        png = urllib.request.urlopen(f"{base}/api/page/{rep['id']}/0.png").read()
        assert png[:4] == b"\x89PNG"
        cert = urllib.request.urlopen(f"{base}/api/cert/{rep['id']}/{proven[0]['id']}").read()
        p = tmp_path / "cert.py"
        p.write_bytes(cert)
        out = subprocess.run([sys.executable, str(p)], capture_output=True, text=True)
        assert out.returncode == 0 and "VERIFIED" in out.stdout
    finally:
        srv.shutdown()


def test_cli_scan_and_verify(tmp_path: Path):
    t = make_trial(20002, tmp_path, "grim_mean")
    out = tmp_path / "report"
    r = CliRunner().invoke(app, ["scan", str(t.pdf), "--out", str(out)])
    assert r.exit_code == 0, r.output
    rep = json.loads((out / "report.json").read_text())
    assert rep["counts"]["PROVEN_IMPOSSIBLE"] >= 1
    assert list((out / "certificates").glob("certificate_*.py"))
    r = CliRunner().invoke(app, ["verify", str(out / "certificates")])
    assert r.exit_code == 0, r.output
