"""Runner script: build isolated environments and benchmark external atom mappers.

Benchmarks four external tools against agave_chem's gold reaction set:
  - rxnmapper     (pip install rxnmapper[rdkit])
  - rxnmapper_v2   (git install RXNMapper_v2; needs torch / transformers)
  - localmapper    (conda + pip install localmapper; needs torch / dgl / dgllife)
  - chython-rxnmap (pip install chython-rxnmap "chython[mapping]")

Usage:
    python run_external_benchmarks.py [--reinstall] [--limit N] [--tools rxnmapper rxnmapper_v2 localmapper chython]
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd is not None else None)


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _conda_python(env_dir: Path) -> Path:
    if os.name == "nt":
        return env_dir / "python.exe"
    return env_dir / "bin" / "python"


def _find_conda() -> str:
    for name in ("conda", "mamba", "micromamba"):
        found = shutil.which(name)
        if found:
            return found
    home = Path.home()
    for candidate in (
        home / "miniforge3" / "bin" / "conda",
        home / "mambaforge" / "bin" / "conda",
        home / "miniconda3" / "bin" / "conda",
        home / "anaconda3" / "bin" / "conda",
    ):
        if candidate.exists():
            return str(candidate)
    try:
        result = subprocess.run(
            ["bash", "-i", "-c", "which conda"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    raise FileNotFoundError(
        "Could not find conda/mamba/micromamba. Install one or ensure it is on PATH."
    )


def _build_uv_venv(*, venv_dir: Path, packages: list[str], reinstall: bool) -> None:
    if reinstall and venv_dir.exists():
        shutil.rmtree(venv_dir)
    if not venv_dir.exists():
        _run(["uv", "venv", str(venv_dir)])
    python = _venv_python(venv_dir)
    _run(["uv", "pip", "install", "--python", str(python)] + packages)


def _build_localmapper_conda_env(*, env_dir: Path, reinstall: bool) -> None:
    """
    Build a conda env for LocalMapper.

    LocalMapper depends on PyTorch, DGL, and DGLLife.  We create a conda
    env for a clean Python and then install the Python stack via pip,
    using the CPU-only PyTorch index so no GPU driver is required.
    DGL 2.x ships standard PyPI wheels (CPU) that work with torch >= 2.0.
    """
    conda = _find_conda()

    if reinstall and env_dir.exists():
        shutil.rmtree(env_dir)

    if not env_dir.exists():
        _run(
            [
                conda,
                "create",
                "--prefix",
                str(env_dir),
                "python=3.10",
                "-y",
            ]
        )

    python = _conda_python(env_dir)

    # Install CPU-only PyTorch first so pip resolves DGL against the right ABI.
    # Pin torch==2.2.1 because DGL 2.1.0's graphbolt C++ library only ships
    # builds for torch 2.0.0–2.2.1.
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "torch==2.2.1",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ]
    )

    # DGL 2.x CPU wheel (available on PyPI for torch >= 2.0)
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "dgl",
            "-f",
            "https://data.dgl.ai/wheels/repo.html",
        ]
    )

    # Remaining Python deps
    # Pin setuptools<80 because setuptools 80+ removed pkg_resources,
    # which localmapper imports.
    # Pin torchdata==0.7.0 because it's the version paired with torch 2.2.x;
    # newer torchdata expects torch.utils._import_utils which doesn't exist
    # in torch 2.2.1.
    # Pin numpy<2 because torch 2.2.1 was compiled against numpy 1.x.
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "setuptools<80",
            "torchdata==0.7.0",
            "numpy<2",
            "pyyaml",
            "pydantic",
            "dgllife",
            "localmapper",
            "rdkit",
            "networkx",
        ]
    )


def _run_benchmark(
    *,
    python: Path,
    bench_dir: Path,
    script_name: str,
    extra_args: list[str] | None = None,
) -> None:
    """Copy the benchmark script to a temp dir and run it there to avoid import conflicts."""
    script_path = bench_dir / script_name
    with tempfile.TemporaryDirectory(prefix="agave_bench_") as d:
        tmpdir = Path(d)
        local_script = tmpdir / script_name
        shutil.copy2(script_path, local_script)
        shutil.copy2(bench_dir / "_bench_utils.py", tmpdir / "_bench_utils.py")

        env = os.environ.copy()
        env["AGAVE_BENCH_DIR"] = str(bench_dir)

        cmd = [str(python), str(local_script)] + (extra_args or [])
        _run(cmd, cwd=tmpdir, env=env)


# ---------------------------------------------------------------------------
# Per-tool build + run
# ---------------------------------------------------------------------------


def _build_rxnmapper_v2_venv(*, venv_dir: Path, reinstall: bool) -> None:
    """
    Build a uv venv for RXNMapper v2.

    RXNMapper v2 depends on torch and transformers with specific pins.
    We install CPU-only PyTorch first so the package resolver picks the
    correct wheel, then install the git package with all its dependencies.
    """
    if reinstall and venv_dir.exists():
        shutil.rmtree(venv_dir)
    if not venv_dir.exists():
        # Pin Python 3.11 because RXNMapper v2 requires numpy<1.24, which has
        # no prebuilt wheels for Python 3.12+ (distutils was removed).
        _run(["uv", "venv", "--python", "3.11", str(venv_dir)])
    python = _venv_python(venv_dir)

    # CPU-only PyTorch first (avoids pulling CUDA wheels).
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "torch",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ]
    )

    # RXNMapper v2 + benchmark deps.
    # Pin setuptools<80 because RXNMapper v2's pyproject.toml requires it
    # and setuptools 80+ removed pkg_resources.
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "git+https://github.com/yvsgrndjn/RXNMapper_v2.git",
            "networkx",
            "setuptools<80",
        ]
    )


def _bench_rxnmapper(
    *, repo_root: Path, bench_dir: Path, reinstall: bool, extra_args: list[str]
) -> None:
    print("\n" + "=" * 60)
    print("=== rxnmapper ===")
    print("=" * 60)
    venv_dir = (repo_root / ".venv-rxnmapper").resolve()
    _build_uv_venv(
        venv_dir=venv_dir,
        packages=["rxnmapper[rdkit]", "setuptools<80", "networkx"],
        reinstall=reinstall,
    )
    _run_benchmark(
        python=_venv_python(venv_dir),
        bench_dir=bench_dir,
        script_name="benchmark_rxnmapper.py",
        extra_args=extra_args,
    )


def _bench_rxnmapper_v2(
    *, repo_root: Path, bench_dir: Path, reinstall: bool, extra_args: list[str]
) -> None:
    print("\n" + "=" * 60)
    print("=== rxnmapper_v2 ===")
    print("=" * 60)
    venv_dir = (repo_root / ".venv-rxnmapper-v2").resolve()
    _build_rxnmapper_v2_venv(venv_dir=venv_dir, reinstall=reinstall)
    _run_benchmark(
        python=_venv_python(venv_dir),
        bench_dir=bench_dir,
        script_name="benchmark_rxnmapper_v2.py",
        extra_args=extra_args,
    )


def _bench_localmapper(
    *, repo_root: Path, bench_dir: Path, reinstall: bool, extra_args: list[str]
) -> None:
    print("\n" + "=" * 60)
    print("=== LocalMapper ===")
    print("=" * 60)
    env_dir = (repo_root / ".conda-localmapper").resolve()
    _build_localmapper_conda_env(env_dir=env_dir, reinstall=reinstall)
    _run_benchmark(
        python=_conda_python(env_dir),
        bench_dir=bench_dir,
        script_name="benchmark_localmapper.py",
        extra_args=extra_args,
    )


def _bench_chython(
    *, repo_root: Path, bench_dir: Path, reinstall: bool, extra_args: list[str]
) -> None:
    print("\n" + "=" * 60)
    print("=== chython-rxnmap ===")
    print("=" * 60)
    venv_dir = (repo_root / ".venv-chython").resolve()
    _build_uv_venv(
        venv_dir=venv_dir,
        packages=["chython-rxnmap", "chython[mapping]", "rdkit", "networkx"],
        reinstall=reinstall,
    )
    _run_benchmark(
        python=_venv_python(venv_dir),
        bench_dir=bench_dir,
        script_name="benchmark_chython.py",
        extra_args=extra_args,
    )


def _bench_agave_chem(
    *, repo_root: Path, bench_dir: Path, reinstall: bool, extra_args: list[str]
) -> None:
    print("\n" + "=" * 60)
    print("=== agave_chem ===")
    print("=" * 60)
    venv_dir = (repo_root / ".venv-agave-chem").resolve()
    _build_uv_venv(
        venv_dir=venv_dir,
        packages=[str(repo_root)],
        reinstall=reinstall,
    )
    _run_benchmark(
        python=_venv_python(venv_dir),
        bench_dir=bench_dir,
        script_name="benchmark_agave_chem.py",
        extra_args=extra_args,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_TOOL_RUNNERS = {
    "rxnmapper": _bench_rxnmapper,
    "rxnmapper_v2": _bench_rxnmapper_v2,
    "localmapper": _bench_localmapper,
    "chython": _bench_chython,
    "agave_chem": _bench_agave_chem,
}

_ALL_TOOLS = list(_TOOL_RUNNERS)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build environments and benchmark external atom mappers"
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        choices=_ALL_TOOLS,
        default=_ALL_TOOLS,
        help="Which tools to benchmark (default: all)",
    )
    parser.add_argument(
        "--gold-reactions",
        default=None,
        help="Path to gold reactions file (default: gold_reactions_filtered.txt next to this script)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save per-tool mapped reaction output files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of reactions processed per tool",
    )
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="Delete and recreate environments before installing",
    )
    parser.add_argument(
        "--agave-mappers",
        nargs="+",
        default=None,
        help="Mapper(s) to pass to benchmark_agave_chem.py (e.g. neural template pipeline)",
    )
    args = parser.parse_args()

    bench_dir = Path(__file__).resolve().parent
    repo_root = bench_dir.parent.parent

    shared_args: list[str] = []
    if args.gold_reactions:
        shared_args += ["--gold-reactions", args.gold_reactions]
    if args.limit is not None:
        shared_args += ["--limit", str(args.limit)]

    for tool in args.tools:
        extra = list(shared_args)
        if tool == "agave_chem" and args.agave_mappers:
            extra += ["--mapper"] + args.agave_mappers
        if args.output_dir:
            if tool == "agave_chem":
                output_path = Path(args.output_dir) / "agave_chem"
                extra += ["--output-prefix", str(output_path)]
            else:
                output_path = Path(args.output_dir) / f"{tool}_results.txt"
                extra += ["--output", str(output_path)]

        _TOOL_RUNNERS[tool](
            repo_root=repo_root,
            bench_dir=bench_dir,
            reinstall=args.reinstall,
            extra_args=extra,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
