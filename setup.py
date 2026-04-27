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
import platform
import subprocess
import sys
from pathlib import Path


def pip(venv, *args):
    is_win = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe)] + list(args), check=True)


def python_exe_in_venv(venv):
    is_win = platform.system() == "Windows"
    return venv / ("Scripts/python.exe" if is_win else "bin/python")


def setup(python_exe, ext_dir, gpu_sm):
    venv = ext_dir / "venv"
    is_win = platform.system() == "Windows"

    print("[setup] Creating venv at %s ..." % venv)
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)


    # ------------------------------------------------------------------ #
    # PyTorch
    # ------------------------------------------------------------------ #
    if gpu_sm >= 100:
        torch_index = "https://download.pytorch.org/whl/cu128"
        torch_pkgs = ["torch>=2.7.0", "torchvision>=0.22.0"]
        print("[setup] SM %d (Blackwell) -> PyTorch 2.7 + CUDA 12.8" % gpu_sm)
    elif gpu_sm >= 70:
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs = ["torch==2.5.1", "torchvision==0.20.1"]
        print("[setup] SM %d -> PyTorch 2.5.1 + CUDA 12.4" % gpu_sm)
    else:
        torch_index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs = ["torch==2.5.1", "torchvision==0.20.1"]
        print("[setup] SM %d (legacy) -> PyTorch 2.5.1 + CUDA 11.8" % gpu_sm)

    print("[setup] Installing PyTorch...")
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # ------------------------------------------------------------------ #
    # Clone Hunyuan3D-2 repo and install hy3dgen package
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

    print("[setup] Installing hy3dgen package...")
    venv_python = python_exe_in_venv(venv)
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-e", str(repo_dir)],
        check=True
    )

    # ------------------------------------------------------------------ #
    # Core dependencies from requirements.txt
    # (shape-only - skip texture custom rasterizer compilation)
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
