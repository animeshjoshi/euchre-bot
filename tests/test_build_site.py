import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
build_site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_site)


class BuildSiteTest(unittest.TestCase):
    def build(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return build_site.build(Path(tmp.name) / "dist")

    def test_site_has_everything_the_browser_loads(self):
        out = self.build()
        for name in ("index.html", "app.js", "worker.js", ".nojekyll", "py/euchre.zip", "py/euchre_q.npz"):
            self.assertTrue((out / name).exists(), name)

    def test_zip_holds_the_package_and_no_junk(self):
        out = self.build()
        names = zipfile.ZipFile(out / "py" / "euchre.zip").namelist()
        for need in ("euchre/__init__.py", "euchre/engine.py", "euchre/webgame.py", "euchre/webbridge.py"):
            self.assertIn(need, names)
        self.assertFalse([n for n in names if "__pycache__" in n or not n.endswith(".py")])

    def test_worker_gets_a_build_hash(self):
        out = self.build()
        self.assertNotIn(build_site.PLACEHOLDER, (out / "worker.js").read_text())

    def test_same_code_gives_the_same_site(self):
        a, b = self.build(), self.build()
        self.assertEqual((a / "py" / "euchre.zip").read_bytes(), (b / "py" / "euchre.zip").read_bytes())
        self.assertEqual((a / "worker.js").read_text(), (b / "worker.js").read_text())


if __name__ == "__main__":
    unittest.main()
