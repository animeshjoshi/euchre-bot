"""Build the static site that GitHub Pages serves.

    python scripts/build_site.py            # writes ./dist
    python -m http.server -d dist 8000      # then open http://localhost:8000

dist/ holds the web page (web/), the bot's Python code zipped up (py/euchre.zip)
and the trained model (py/euchre_q.npz). The browser runs the zip with Pyodide.
"""
import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER = "__BUILD__"


def zip_package(src_dir, dest):
    """Zip euchre/*.py the same way every time, so the same code gives the same bytes."""
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src_dir.rglob("*.py")):
            info = zipfile.ZipInfo(f"{src_dir.name}/{f.relative_to(src_dir).as_posix()}", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, f.read_bytes())


def build(out):
    out = Path(out)
    model = ROOT / "models" / "euchre_q.npz"
    if not model.exists():
        sys.exit(f"missing {model}: the site would fall back to the weaker rule-based card player")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "web", out)
    (out / "py").mkdir()
    zip_package(ROOT / "euchre", out / "py" / "euchre.zip")
    shutil.copy(model, out / "py" / "euchre_q.npz")

    # Cache-busting: the worker asks for py/* with ?v=<hash of what's inside>, so after a deploy
    # a browser can't pair a new worker.js with a stale cached zip or model.
    h = hashlib.sha256()
    for name in ("euchre.zip", "euchre_q.npz"):
        h.update((out / "py" / name).read_bytes())
    worker = out / "worker.js"
    worker.write_text(worker.read_text().replace(PLACEHOLDER, h.hexdigest()[:12]))

    (out / ".nojekyll").write_text("")  # serve files as they are; skip GitHub's Jekyll step
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "dist"))
    print("built", build(ap.parse_args().out))
