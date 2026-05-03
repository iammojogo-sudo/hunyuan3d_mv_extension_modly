"""
Hunyuan3D-2mv - Modly extension setup script.

Called by Modly at install time:
    python setup.py <json_args>

json_args keys:
    python_exe  - path to Modly's embedded Python
    ext_dir     - absolute path to this extension directory
    gpu_sm      - GPU compute capability as integer (e.g. 89 for RTX 4050)
"""
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


IS_WIN = platform.system() == "Windows"


def pip(venv, *args):
    pip_exe = venv / ("Scripts/pip.exe" if IS_WIN else "bin/pip")
    subprocess.run([str(pip_exe)] + list(args), check=True)


def python_exe_in_venv(venv):
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def _resolve_cuda_home():
    """Find the CUDA toolkit root via every known Windows mechanism."""
    for k in ("CUDA_HOME", "CUDA_PATH"):
        v = os.environ.get(k)
        if v and Path(v).exists():
            return v
    for k, v in os.environ.items():
        if k.startswith("CUDA_PATH_V") and v and Path(v).exists():
            return v
    nvcc = shutil.which("nvcc") or shutil.which("nvcc.exe")
    if nvcc:
        root = str(Path(nvcc).parent.parent)
        if Path(root).exists():
            return root
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA")
        versions = []
        for i in range(winreg.QueryInfoKey(key)[0]):
            try:
                sk = winreg.OpenKey(key, winreg.EnumKey(key, i))
                p, _ = winreg.QueryValueEx(sk, "InstallDir")
                if p and Path(p).exists():
                    versions.append(p)
            except OSError:
                pass
        if versions:
            return versions[-1]
    except OSError:
        pass
    cuda_base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if cuda_base.exists():
        dirs = sorted([d for d in cuda_base.iterdir() if d.is_dir()], reverse=True)
        if dirs:
            return str(dirs[0])
    return None


def _find_cl_exe():
    """Try to locate cl.exe (MSVC compiler) for the rasterizer build on Windows."""
    candidates = []
    for base in [
        r"C:\Program Files\Microsoft Visual Studio",
        r"C:\Program Files (x86)\Microsoft Visual Studio",
    ]:
        base_p = Path(base)
        if base_p.exists():
            for cl in base_p.rglob("cl.exe"):
                if "x64" in str(cl) or "amd64" in str(cl).lower():
                    candidates.append(cl)
    return candidates[0] if candidates else None


def _build_custom_rasterizer(venv_python, rast_dir):
    """
    Build the custom_rasterizer C extension in-place and copy it to site-packages.

    On Windows:
      - ninja must be on PATH (installed before this call).
      - MSVC cl.exe must be reachable; we try to locate it automatically.
      - After build, the .pyd is copied to site-packages so the bare
        `import custom_rasterizer` resolves regardless of cwd / sys.path.
    """
    rast_dir = Path(rast_dir)

    # Check for a pre-built .pyd shipped with the extension repo
    ext_dir = Path(__file__).parent
    prebuilt = (
        list(ext_dir.glob("custom_rasterizer_kernel*.pyd")) +
        list(ext_dir.glob("custom_rasterizer_kernel*.so"))
    )
    if prebuilt:
        artifact = prebuilt[0]
        print("[setup] Found pre-built rasterizer: %s" % artifact)
        try:
            if IS_WIN:
                site_pkgs = venv_python.parent.parent / "Lib" / "site-packages"
            else:
                site_pkgs = sorted(
                    (venv_python.parent.parent / "lib").glob("python*/site-packages")
                )[-1]
            dest = site_pkgs / artifact.name
            shutil.copy2(str(artifact), str(dest))
            print("[setup] Installed pre-built %s -> %s" % (artifact.name, site_pkgs))
        except Exception as exc:
            print("[setup] Could not copy pre-built rasterizer: %s" % exc)
        return True

    print("[setup] Building custom_rasterizer in %s ..." % rast_dir)

    env = os.environ.copy()

    # Inject CUDA_HOME so torch's cpp_extension finds the toolkit
    cuda_home = _resolve_cuda_home()
    if cuda_home:
        env["CUDA_HOME"] = cuda_home
        env["CUDA_PATH"] = cuda_home
        print("[setup] CUDA_HOME resolved: %s" % cuda_home)
    else:
        print("[setup] WARNING: Could not auto-detect CUDA_HOME.")

    if IS_WIN:
        # Ensure venv Scripts (ninja, etc.) are on PATH
        venv_scripts = venv_python.parent
        env["PATH"] = str(venv_scripts) + os.pathsep + env.get("PATH", "")

        # Inject MSVC cl.exe if not already visible
        if shutil.which("cl") is None:
            cl = _find_cl_exe()
            if cl:
                env["PATH"] = str(cl.parent) + os.pathsep + env["PATH"]
                print("[setup] Found cl.exe: %s" % cl)
            else:
                print(
                    "[setup] WARNING: cl.exe not found on PATH.\n"
                    "[setup]   Install 'Desktop development with C++' in Visual Studio,\n"
                    "[setup]   or run this setup from a VS Developer Command Prompt."
                )

    result = subprocess.run(
        [str(venv_python), "setup.py", "build_ext", "--inplace"],
        cwd=str(rast_dir),
        env=env,
    )

    if result.returncode != 0:
        print(
            "[setup] WARNING: custom_rasterizer build exited with code %d.\n"
            "[setup]   Texture generation will fail until this is fixed." % result.returncode
        )
        return False

    # Find the compiled artifact (the kernel module name is custom_rasterizer_kernel)
    built = (
        list(rast_dir.glob("custom_rasterizer_kernel*.pyd")) +
        list(rast_dir.glob("custom_rasterizer_kernel*.so"))
    )
    if not built:
        print("[setup] WARNING: build reported success but no .pyd/.so found in %s." % rast_dir)
        return False

    artifact = built[0]
    print("[setup] custom_rasterizer built: %s" % artifact)

    # Copy to venv site-packages so bare `import custom_rasterizer` always works
    try:
        if IS_WIN:
            site_pkgs = venv_python.parent.parent / "Lib" / "site-packages"
        else:
            site_pkgs = sorted(
                (venv_python.parent.parent / "lib").glob("python*/site-packages")
            )[-1]

        dest = site_pkgs / artifact.name
        shutil.copy2(str(artifact), str(dest))
        print("[setup] Installed %s -> %s" % (artifact.name, site_pkgs))
    except Exception as exc:
        print("[setup] Note: could not copy rasterizer to site-packages (%s)." % exc)
        print("[setup]   The extension dir must stay on sys.path at runtime.")

    return True


def setup(python_exe, ext_dir, gpu_sm):
    venv = ext_dir / "venv"

    print("[setup] Creating venv at %s ..." % venv)
    subprocess.run([str(python_exe), "-m", "venv", str(venv)], check=True)

    venv_python = python_exe_in_venv(venv)

    # ------------------------------------------------------------------ #
    # Build prerequisites — ninja first so the rasterizer can find it
    # ------------------------------------------------------------------ #
    print("[setup] Installing build prerequisites (ninja, setuptools, wheel)...")
    pip(venv, "install", "ninja", "setuptools", "wheel")

    # ------------------------------------------------------------------ #
    # PyTorch
    # ------------------------------------------------------------------ #
    if gpu_sm >= 100:
        torch_index = "https://download.pytorch.org/whl/cu128"
        torch_pkgs = ["torch>=2.7.0", "torchvision>=0.22.0", "torchaudio>=2.7.0"]
        print("[setup] SM %d (Blackwell) -> PyTorch 2.7 + CUDA 12.8" % gpu_sm)
    elif gpu_sm >= 70:
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs = ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
        print("[setup] SM %d -> PyTorch 2.5.1 + CUDA 12.4" % gpu_sm)
    else:
        torch_index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs = ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
        print("[setup] SM %d (legacy) -> PyTorch 2.5.1 + CUDA 11.8" % gpu_sm)

    print("[setup] Installing PyTorch...")
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # ------------------------------------------------------------------ #
    # xformers  (the Triton warning at runtime is harmless on Windows)
    # ------------------------------------------------------------------ #
    print("[setup] Installing xformers...")
    if gpu_sm >= 70:
        pip(venv, "install", "xformers==0.0.28.post3", "--index-url", torch_index)
    else:
        pip(venv, "install", "xformers==0.0.28.post2", "--index-url",
            "https://download.pytorch.org/whl/cu118")

    # ------------------------------------------------------------------ #
    # Core dependencies
    # ------------------------------------------------------------------ #
    print("[setup] Installing core dependencies...")
    pip(venv, "install",
        "transformers==4.40.2",
        "diffusers==0.27.2",
        "huggingface_hub==0.23.5",
        "accelerate",
        "omegaconf",
        "einops",
        "Pillow",
        "numpy",
        "scipy",
        "trimesh",
        "pymeshlab",
        "pygltflib",
        "opencv-python-headless",
        "tqdm",
        "safetensors",
        "rembg",
    )

    # triton: Linux-only; skip silently on Windows (xformers will warn but still work)
    if not IS_WIN:
        try:
            pip(venv, "install", "triton")
        except subprocess.CalledProcessError:
            print("[setup] triton not available — skipping (non-fatal).")

    # ------------------------------------------------------------------ #
    # onnxruntime
    # ------------------------------------------------------------------ #
    if gpu_sm >= 70:
        print("[setup] Installing onnxruntime-gpu...")
        try:
            pip(venv, "install", "onnxruntime-gpu")
        except subprocess.CalledProcessError:
            print("[setup] onnxruntime-gpu failed, falling back to cpu.")
            pip(venv, "install", "onnxruntime")
    else:
        pip(venv, "install", "onnxruntime")

    # ------------------------------------------------------------------ #
    # Clone Hunyuan3D-2 repo
    # ------------------------------------------------------------------ #
    repo_dir = ext_dir / "Hunyuan3D-2"
    if not repo_dir.exists():
        print("[setup] Cloning Hunyuan3D-2 repo...")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git",
             str(repo_dir)],
            check=True
        )
    else:
        print("[setup] Repo already exists, skipping clone.")

    # ------------------------------------------------------------------ #
    # Build custom_rasterizer BEFORE installing the package
    # ------------------------------------------------------------------ #
    rast_dir = repo_dir / "hy3dgen" / "texgen" / "custom_rasterizer"
    rast_ok = _build_custom_rasterizer(venv_python, rast_dir)
    if not rast_ok:
        print(
            "[setup] *** custom_rasterizer was NOT built. ***\n"
            "[setup]     Texture generation will fail until this is resolved.\n"
            "[setup]     Fix the compiler error above then reinstall the extension."
        )

    # ------------------------------------------------------------------ #
    # Install hy3dgen package (editable) — rasterizer must be built first
    # ------------------------------------------------------------------ #
    print("[setup] Installing hy3dgen package...")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-e", str(repo_dir)],
        check=True
    )

    # hy3dgen pulls in newer huggingface_hub which breaks diffusers==0.27.2
    # (cached_download was removed). Force-reinstall the pinned versions last.
    print("[setup] Re-pinning huggingface_hub and diffusers versions...")
    pip(venv, "install", "--force-reinstall", "--no-deps",
        "huggingface_hub==0.23.5",
        "diffusers==0.27.2",
        "transformers==4.40.2",
    )

    # ------------------------------------------------------------------ #
    # Final import verification
    # ------------------------------------------------------------------ #
    print("[setup] Verifying custom_rasterizer import...")
    check = subprocess.run(
        [str(venv_python), "-c",
         "import custom_rasterizer_kernel; import custom_rasterizer; print('custom_rasterizer: OK')"],
        capture_output=True, text=True,
    )
    if "OK" in check.stdout:
        print("[setup] %s" % check.stdout.strip())
    else:
        stderr = check.stderr.strip()
        print(
            "[setup] custom_rasterizer import FAILED.\n"
            "[setup]   %s\n"
            "[setup]   Ensure MSVC (Visual Studio C++ build tools) and the CUDA\n"
            "[setup]   toolkit matching your PyTorch build are installed, then\n"
            "[setup]   reinstall this extension." % stderr
        )

    print("[setup] Done. Venv ready at: %s" % venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(
            python_exe=sys.argv[1],
            ext_dir=Path(sys.argv[2]),
            gpu_sm=int(sys.argv[3]),
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe=args["python_exe"],
            ext_dir=Path(args["ext_dir"]),
            gpu_sm=int(args["gpu_sm"]),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":89}\'')
        sys.exit(1)
