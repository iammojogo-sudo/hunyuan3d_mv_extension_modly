import os
import sys
from pathlib import Path
from threading import Event, Thread
from typing import Optional, Callable, List, Tuple
from PIL import Image
from trimesh import Trimesh
from py3dgen import Py3DGen

class Hunyuan3DGenerator:
    def __init__(self, inputs_dir: Path, outputs_dir: Path):
        self.inputs_dir = inputs_dir
        self.outputs_dir = outputs_dir
        self.py3dgen = Py3DGen()

    def load_model(self) -> Optional[Py3DGen]:
        """Load the 3D model from file"""
        if self._model is None:
            try:
                # Load model from file
                self._model = Py3DGen(str(self.model_path))
            except Exception as e:
                print(f"Error loading model: {e}")
                return None
        return self._model

    def generate_shape(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """Generate a 3D shape using the Hunyuan model"""
        # Extract parameters
        node_id = self.model_path.name.split('.')[0]
        default_variant = _SHAPE_SUBFOLDERS.get(node_id, "hunyuan3d-dit-v2-mv-turbo")
        variant = params.get("model_variant", default_variant)
        steps = _safe_int(params.get("num_inference_steps"), 30)
        octree_res = _safe_int(params.get("octree_resolution"), 380)
        seed = _safe_int(params.get("seed"), -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        guidance_scale = _safe_float(params.get("guidance_scale"), 5.0)
        num_chunks = _safe_int(params.get("num_chunks"), 8000)
        box_v = _safe_float(params.get("box_v"), 1.01)
        mc_level = _safe_float(params.get("mc_level"), 0.0)

        # Preprocess front image
        self._report(progress_cb, 5, "Preprocessing front view...")
        front_image = self._preprocess(image_bytes)
        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled()

        # Extract other views
        images = []
        for pct, view_name in ((10, "left"), (14, "back"), (18, "right")):
            image = self._optional_view_image(params, view_name)
            if image is not None:
                self._report(progress_cb, pct, f"Preprocessing {view_name} view...")
                images.append(image)
                if cancel_event and cancel_event.is_set():
                    raise GenerationCancelled()

        # Generate 3D shape
        self._report(progress_cb, 30, "Generating mesh...")
        stop_evt = threading.Event()
        progress_thread = None
        if progress_cb:
            progress_thread = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 30, 92, "Generating mesh...", stop_evt),
                daemon=True,
            )
            progress_thread.start()

        try:
            # Load model and generate mesh
            self.load_model()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                result = self._model(
                    image=images,
                    num_inference_steps=steps,
                    octree_resolution=octree_res,
                    guidance_scale=guidance_scale,
                    num_chunks=num_chunks,
                    box_v=box_v,
                    mc_level=mc_level,
                    generator=generator,
                    output_type="trimesh",
                )
                print(f"Pipeline result type: {type(result)}")
                mesh = result[0] if isinstance(result, (list, tuple)) else result
                print(f"Mesh type: {type(mesh)}")
        finally:
            stop_evt.set()
            if progress_thread:
                progress_thread.join(timeout=1.0)

        # Validate and export mesh
        self._report(progress_cb, 94, "Validating and exporting mesh...")
        if mesh is None:
            raise RuntimeError("Generated mesh is None")
        if not hasattr(mesh, "vertices") or mesh.vertices is None or len(mesh.vertices) == 0:
            raise RuntimeError("Generated mesh has no vertices")
        if not hasattr(mesh, "faces") or mesh.faces is None or len(mesh.faces) == 0:
            raise RuntimeError("Generated mesh has no faces")

        # Export GLB
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.outputs_dir / ("%d_%s.glb" % (int(time.time()), uuid.uuid4().hex[:8]))
        mesh.export(str(out_path))
        print(f"Exported GLB to: {out_path}")

    def generate_texture(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """Generate a textured 3D mesh"""
        # Load input mesh
        mesh_path = params.get("mesh_path") or params.get("input_path")
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh not found at {mesh_path}")
        mesh = trimesh.load(mesh_path)

        # Preprocess front image
        self._report(progress_cb, 5, "Preprocessing front view...")
        front_image = self._preprocess(image_bytes)
        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled()

        # Extract other views
        images = []
        for pct, view_name in ((15, "left"), (26, "back"), (30, "right")):
            image = self._optional_view_image(params, view_name)
            if image is not None:
                self._report(progress_cb, pct, f"Preprocessing {view_name} view...")
                images.append(image)
                if cancel_event and cancel_event.is_set():
                    raise GenerationCancelled()

        # Generate textured mesh
        self._report(progress_cb, 38, "Generating textures...")
        stop_evt = threading.Event()
        progress_thread = None
        if progress_cb:
            progress_thread = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 38, 88, "Generating textures...", stop_evt),
                daemon=True,
            )
            progress_thread.start()

        try:
            # Load model and generate textured mesh
            self.load_model()
            with torch.no_grad():
                textured_mesh = self.py3dgen(mesh, image=images)
                print(f"Texturing complete")
        finally:
            stop_evt.set()
            if progress_thread:
                progress_thread.join(timeout=1.0)

        # Validate and export textured mesh
        self._report(progress_cb, 90, "Validating and exporting textured mesh...")
        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled()

        # Export GLB
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.outputs_dir / ("%d_%s_textured.glb" % (int(time.time()), uuid.uuid4().hex[:8]))
        textured_mesh.export(str(out_path))
        print(f"Exported textured GLB to: {out_path}")

    def _report(self, progress_cb: Optional[Callable[[int, str], None]], pct: int, message: str) -> None:
        """Report progress"""
        if progress_cb:
            progress_cb(pct, message)

def smooth_progress(progress_cb: Callable[[int, str], None], start_pct: int, end_pct: int, message: str, stop_evt: Event):
    """Smoothly report progress"""
    for pct in range(start_pct, end_pct + 1):
        if stop_evt.is_set():
            break
        try:
            progress_cb(pct, message)
        except Exception as e:
            print(f"Error reporting progress: {e}")
            break

def main():
    # Parse command-line arguments
    inputs_dir = Path(sys.argv[1])
    outputs_dir = Path(sys.argv[2])

    # Create generator instance
    generator = Hunyuan3DGenerator(inputs_dir, outputs_dir)

    # Load model and generate 3D shape or textured mesh
    image_bytes = ...  # load image bytes from file or other source
    params = {...}  # define parameters for generating 3D shape or textured mesh

    if generator._is_texture_node:
        out_path = generator.generate_texture(image_bytes, params)
    else:
        out_path = generator.generate_shape(image_bytes, params)

    print(f"Generated output: {out_path}")

if __name__ == "__main__":
    main()
