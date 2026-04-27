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


def build_extension(venv_python, source_dir, label):
    """Build and install a C++/CUDA extension from its setup.py."""
    source_dir = Path(source_dir)
    if not source_dir.exists():
        print("[setup] %s source not found at %s — skipping." % (label, source_dir))
        return False
    print("[setup] Building %s..." % label)
    try:
        subprocess.run(
            [str(venv_python), "setup.py", "install"],
            cwd=str(source_dir),
            check=True,
        )
        print("[setup] %s built and installed successfully." % label)
        return True
    except subprocess.CalledProcessError as e:
        print(
            "[setup] WARNING: %s build failed (exit code %d).\n"
            "  Make sure Visual Studio C++ Build Tools are installed:\n"
            "    https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
            "  Then run manually:\n"
            "    cd \"%s\"\n"
            "    python setup.py install" % (label, e.returncode, source_dir)
        )
        return False
    except Exception as e:
        print("[setup] WARNING: %s build error: %s" % (label, e))
        return False


def setup(python_exe, ext_dir, gpu_sm):
    venv = ext_dir / "venv"

    print("[setup] Creating venv at %s ..." % venv)
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

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
    # xformers
    # ------------------------------------------------------------------ #
    print("[setup] Installing xformers...")
    if gpu_sm >= 70:
        pip(venv, "install", "xformers==0.0.28.post3", "--index-url", torch_index)
    else:
        pip(venv, "install", "xformers==0.0.28.post2", "--index-url",
            "https://download.pytorch.org/whl/cu118")

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
        "onnxruntime",
        "pybind11",
    )

    # ------------------------------------------------------------------ #
    # nvdiffrast — prebuilt NVIDIA rasteriser (replaces custom_rasterizer)
    #
    # nvdiffrast ships compiled wheels via pip so users do NOT need the
    # CUDA toolkit or Visual Studio Build Tools installed. generator.py
    # patches MeshRender at runtime to use nvdiffrast instead of the
    # custom_rasterizer CUDA kernel, giving everyone working texturing
    # out of the box.
    # ------------------------------------------------------------------ #
    print("[setup] Installing nvdiffrast...")
    try:
        pip(venv, "install", "nvdiffrast")
        print("[setup] nvdiffrast installed successfully.")
    except subprocess.CalledProcessError:
        print(
            "[setup] WARNING: nvdiffrast failed to install.
"
            "  Texture generation will not work.
"
            "  Try manually: pip install nvdiffrast"
        )

    # ------------------------------------------------------------------ #
    # onnxruntime-gpu if supported
    # ------------------------------------------------------------------ #
    if gpu_sm >= 70:
        print("[setup] Installing onnxruntime-gpu...")
        try:
            pip(venv, "install", "onnxruntime-gpu")
        except subprocess.CalledProcessError:
            print("[setup] onnxruntime-gpu failed, falling back to cpu.")
            pip(venv, "install", "onnxruntime")

    # ------------------------------------------------------------------ #
    # Optional C++ extension: mesh_processor
    #
    # mesh_processor accelerates UV inpainting inside MeshRender.
    # A pure-Python fallback already exists in the repo, so this build
    # is best-effort — texturing works without it, just slightly slower.
    #
    # Requires: Visual Studio C++ Build Tools (Windows)
    #   https://visualstudio.microsoft.com/visual-cpp-build-tools/
    #
    # NOTE: custom_rasterizer is NO LONGER built here. nvdiffrast (above)
    # provides the same functionality without needing the CUDA toolkit.
    # ------------------------------------------------------------------ #
    texgen_dir = repo_dir / "hy3dgen" / "texgen"

    build_extension(
        venv_python,
        texgen_dir / "differentiable_renderer",
        "mesh_processor (C++ / pybind11, optional)",
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
