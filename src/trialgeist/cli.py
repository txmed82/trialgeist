"""trialgeist command line."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Machine-verified mathematical forensics for clinical trials.")


@app.command()
def scan(pdf: Path = typer.Argument(..., help="trial PDF, PMC JATS .nxml/.xml, or claims .json"),
         nct: str = typer.Option(None, help="ClinicalTrials.gov ID (NCT########)"),
         out: Path = typer.Option(Path("report"), help="output directory"),
         km_eps_pt: float = typer.Option(1.0, help="KM plotting tolerance in PDF points")):
    """Scan a trial PDF (or PMC XML) and write report.md, report.json and certificates/."""
    from .scan import scan_any, write_report

    rep = scan_any(pdf, nct=nct, km_eps_pt=km_eps_pt) if str(pdf).lower().endswith(".pdf") else scan_any(pdf, nct=nct)
    write_report(rep, out)
    typer.echo((out / "report.md").read_text().split("\n## ")[0])


@app.command()
def check(claims: Path, out: Path = typer.Option(Path("report"))):
    """Check a structured claims JSON file (see claims.py for the schema)."""
    from .scan import scan_claims, write_report

    rep = scan_claims(claims)
    write_report(rep, out)
    typer.echo((out / "report.md").read_text().split("\n## ")[0])


@app.command()
def verify(certdir: Path):
    """Re-run every certificate in a directory."""
    from .certificates import run_certificate

    bad = 0
    for p in sorted(certdir.glob("certificate_*.py")):
        ok, dt, outp = run_certificate(p.read_text(), certdir / ".verify", p.stem.removeprefix("certificate_"))
        typer.echo(f"{'OK ' if ok else 'BAD'} {dt:5.2f}s {p.name}")
        bad += not ok
    raise typer.Exit(1 if bad else 0)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765):
    """Local browser studio (no network calls, nothing leaves the machine)."""
    from .server import run

    run(host, port)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
