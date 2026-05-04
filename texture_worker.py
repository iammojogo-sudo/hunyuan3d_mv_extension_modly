"""
texture_worker.py — called by processor.js to apply texture to an untextured GLB.

Usage:
    python texture_worker.py <mesh_path> <params_json> <models_dir> <workspace_dir>

Streams newline-delimited JSON to stdout:
    {"type": "progress", "pct": 0-100, "step": "..."}
    {"type": "log",      "message": "..."}
    {"type": "done",     "output_path": "..."}
    {"type": "error",    "message": "...", "traceback": "..."}

All other output (including from the generator) goes to stderr.
"""
import json
import os
import sys
import traceback
from pathlib import Path


# ------------------------------------------------------------------ #
# Redirect stray print() calls to stderr so stdout stays clean JSON.
# ------------------------------------------------------------------ #
_real_print = print

def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _real_print(*args, **kwargs)


# ------------------------------------------------------------------ #
# Protocol helpers
# ------------------------------------------------------------------ #

def send(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def progress(pct: int, step: str = "") -> None:
    send({"type": "progress", "pct": pct, "step": step})


def log(message: str) -> None:
    send({"type": "log", "message": message})


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> None:
    if len(sys.argv) < 5:
        send({"type": "error",
              "message": "texture_worker.py requires 4 arguments: "
                         "mesh_path params_json models_dir workspace_dir"})
        sys.exit(1)

    mesh_path     = sys.argv[1]
    params_json   = sys.argv[2]
    models_dir    = Path(sys.argv[3])
    workspace_dir = Path(sys.argv[4])
    ext_dir       = Path(os.environ.get("EXTENSION_DIR", Path(__file__).parent))

    # ---- Parse params ------------------------------------------------
    try:
        params = json.loads(params_json)
    except Exception:
        params = {}

    log(f"texture_worker: mesh={mesh_path}")
    log(f"texture_worker: models_dir={models_dir}")
    log(f"texture_worker: workspace_dir={workspace_dir}")
    log(f"texture_worker: ext_dir={ext_dir}")

    # ---- Ensure hy3dgen is on path -----------------------------------
    repo_dir = ext_dir / "Hunyuan3D-2"
    if not repo_dir.exists():
        send({"type": "error",
              "message": f"Hunyuan3D-2 source not found at {repo_dir}. "
                         "Please reinstall or repair the extension."})
        sys.exit(1)

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    if str(ext_dir) not in sys.path:
        sys.path.insert(0, str(ext_dir))

    # ---- Modly api path (for BaseGenerator import) -------------------
    modly_api_dir = os.environ.get("MODLY_API_DIR", "")
    if modly_api_dir and modly_api_dir not in sys.path:
        sys.path.insert(0, modly_api_dir)

    # ---- Load manifest and generator class ---------------------------
    manifest_path = ext_dir / "manifest.json"
    try:
        import importlib.util as _ilu
        manifest   = json.loads(manifest_path.read_text(encoding="utf-8"))
        class_name = manifest["generator_class"]

        spec = _ilu.spec_from_file_location("generator", ext_dir / "generator.py")
        mod  = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        GenClass = getattr(mod, class_name)
    except Exception:
        send({"type": "error",
              "message": "Failed to load generator class",
              "traceback": traceback.format_exc()})
        sys.exit(1)

    # ---- Resolve model_dir -------------------------------------------
    # Paint weights (delight + paint subfolders) are downloaded into
    # the texture node's own model dir: models_dir/hunyuan3d2mv/texture
    # This matches what the registry sets as MODEL_DIR for this node.
    ext_id    = manifest["id"]
    model_dir = models_dir / ext_id / "texture"
    model_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    log(f"texture_worker: model_dir={model_dir}")

    # ---- Instantiate generator ---------------------------------------
    gen = GenClass(model_dir, workspace_dir)

    # Inject texture node manifest fields
    nodes        = manifest.get("nodes", [])
    texture_node = next((n for n in nodes if n.get("id") == "texture"), {})
    gen.hf_repo          = texture_node.get("hf_repo", manifest.get("hf_repo", ""))
    gen.hf_skip_prefixes = texture_node.get("hf_skip_prefixes", [])
    gen.download_check   = texture_node.get("download_check", "")
    gen._params_schema   = texture_node.get("params_schema", [])

    # ---- Load model --------------------------------------------------
    progress(5, "Loading model...")
    try:
        gen.load()
    except Exception:
        send({"type": "error",
              "message": "Failed to load generator",
              "traceback": traceback.format_exc()})
        sys.exit(1)

    # ---- Run texture -------------------------------------------------
    def progress_cb(pct: int, step: str = "") -> None:
        # Remap texture()'s 0-100 into 8-99 to leave room for load step
        mapped = 8 + int(pct * 0.91)
        progress(mapped, step)

    try:
        output_path = gen.texture(
            mesh_path=mesh_path,
            params=params,
            progress_cb=progress_cb,
            cancel_event=None,
        )
    except Exception:
        send({"type": "error",
              "message": "Texture generation failed",
              "traceback": traceback.format_exc()})
        sys.exit(1)

    # ---- Done --------------------------------------------------------
    progress(100, "Done")
    send({"type": "done", "output_path": str(output_path)})


if __name__ == "__main__":
    main()
