"""Standard-library-only runtime setup for application subprocesses."""
import os
from pathlib import Path
import sys


def conda_runtime_env():
    """Prefer this Python environment's C++ runtime for a child process.

    Scope the library path to application children; do not modify the running
    notebook or shell. sys.prefix identifies the kernel's environment even
    when CONDA_PREFIX was inherited from a different Jupyter server environment.
    """
    env = os.environ.copy()
    libdir = Path(sys.prefix) / "lib"
    if not sys.platform.startswith("linux") or not (libdir / "libstdc++.so.6").is_file():
        return env
    current = env.get("LD_LIBRARY_PATH", "")
    first = current.split(":", 1)[0]
    if not first or Path(first).resolve() != libdir.resolve():
        env["LD_LIBRARY_PATH"] = str(libdir) + ((":" + current) if current else "")
    return env


def maybe_reexec_with_conda_libstdcpp():
    """Set the loader path at process startup, before any compiled imports."""
    env = conda_runtime_env()
    if env.get("LD_LIBRARY_PATH") == os.environ.get("LD_LIBRARY_PATH"):
        return
    # Preserve -u, -m and other interpreter flags as well as script arguments.
    argv = getattr(sys, "orig_argv", [sys.executable, *sys.argv])
    os.execve(sys.executable, [sys.executable, *argv[1:]], env)
