"""Regression checks for Linux shared-library loading before compiled imports."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts import runtime_env

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_uses_kernel_prefix_and_preserves_parent_env(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libstdc++.so.6").touch()
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("CONDA_PREFIX", "/different/jupyter/server")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/cuda/lib64:/cluster/lib")
    env = runtime_env.conda_runtime_env()
    assert env["LD_LIBRARY_PATH"] == str(lib) + ":/cuda/lib64:/cluster/lib"
    assert os.environ["LD_LIBRARY_PATH"] == "/cuda/lib64:/cluster/lib"
    monkeypatch.setenv("LD_LIBRARY_PATH", env["LD_LIBRARY_PATH"])
    assert runtime_env.conda_runtime_env() == dict(os.environ)


def test_no_installed_runtime_leaves_environment_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    assert runtime_env.conda_runtime_env() == dict(os.environ)


@pytest.mark.skipif(not sys.platform.startswith("linux") or not shutil.which("gcc"),
                    reason="Requires the Linux loader and gcc for a versioned-library fixture")
def test_reexec_resolves_actual_glibcxx_version_failure(tmp_path):
    # Tiny isolated libraries reproduce the loader failure without installing
    # old system packages or replacing the host's real C++ runtime.
    good = tmp_path / "environment/lib"
    bad = tmp_path / "old_system_lib"
    good.mkdir(parents=True)
    bad.mkdir()
    source = tmp_path / "runtime.c"
    source.write_text("int cfm_runtime_probe(void) { return 29; }\n")
    for folder, version in ((bad, "3.4.28"), (good, "3.4.29")):
        symbols = folder / "symbols.map"
        symbols.write_text(f"GLIBCXX_{version} {{ global: cfm_runtime_probe; local: *; }};\n")
        subprocess.run(["gcc", "-shared", "-fPIC", str(source),
                        f"-Wl,--version-script={symbols}", "-Wl,-soname,libstdc++.so.6",
                        "-o", str(folder / "libstdc++.so.6")], check=True, capture_output=True)
    consumer = tmp_path / "consumer.c"
    consumer.write_text("extern int cfm_runtime_probe(void);\nint cfm_call(void) { return cfm_runtime_probe(); }\n")
    binary = tmp_path / "consumer.so"
    subprocess.run(["gcc", "-shared", "-fPIC", str(consumer), "-L" + str(good),
                    "-l:libstdc++.so.6", "-o", str(binary)], check=True, capture_output=True)
    probe = tmp_path / "probe.py"
    probe.write_text(
        f"import sys\nsys.path.insert(0, {str(ROOT)!r})\nsys.prefix = {str(good.parent)!r}\n"
        "if '--fixed' in sys.argv:\n"
        "    from scripts.runtime_env import maybe_reexec_with_conda_libstdcpp\n"
        "    maybe_reexec_with_conda_libstdcpp()\n"
        f"import ctypes\nassert ctypes.CDLL({str(binary)!r}).cfm_call() == 29\n"
        "print('versioned library loaded')\n"
    )
    env = dict(os.environ, LD_LIBRARY_PATH=str(bad), CONDA_PREFIX="/wrong/server/env")
    broken = subprocess.run([sys.executable, "-u", str(probe)], env=env, capture_output=True, text=True, timeout=30)
    assert broken.returncode != 0
    assert "GLIBCXX_3.4.29" in broken.stderr and "not found" in broken.stderr
    fixed = subprocess.run([sys.executable, "-u", str(probe), "--fixed"], env=env, capture_output=True, text=True, timeout=30)
    assert fixed.returncode == 0, fixed.stderr
    assert "versioned library loaded" in fixed.stdout


@pytest.mark.parametrize("entrypoint", ["run_scrna_drug_transport.py", "update_scrna_baselines.py", "export_scrna_paper.py"])
@pytest.mark.skipif(not sys.platform.startswith("linux") or not shutil.which("g++"),
                    reason="Requires Linux and g++ to locate the host C++ runtime")
def test_cli_bootstrap_precedes_compiled_imports(tmp_path, entrypoint):
    host_runtime = Path(subprocess.check_output(["g++", "-print-file-name=libstdc++.so.6"], text=True).strip())
    if not host_runtime.is_file():
        pytest.skip("No host C++ runtime found")
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libstdc++.so.6").symlink_to(host_runtime.resolve())
    script = ROOT / "scripts" / entrypoint
    # Fail at the first compiled import unless startup selected the interpreter
    # environment. Run the real CLI, including its re-exec and argument parsing.
    launcher = f"""
import builtins, os, runpy, sys
sys.prefix = {str(tmp_path)!r}
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in ('numpy', 'pandas', 'torch', 'scipy'):
        if os.environ.get('LD_LIBRARY_PATH', '').split(':')[0] != {str(lib)!r}:
            raise ImportError('Compiled import occurred before C++ runtime setup')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
sys.argv = [{str(script)!r}, '--help']
runpy.run_path({str(script)!r}, run_name='__main__')
"""
    env = dict(os.environ, CONDA_PREFIX="/wrong/server/env")
    env.pop("LD_LIBRARY_PATH", None)
    result = subprocess.run([sys.executable, "-u", "-c", launcher], env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
