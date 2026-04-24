import io
import os
import sys

# Redirect print to stderr so stdout stays clean for JSON protocol
_print = print
def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)
import time
import uuid
import threading
@@ -25,14 +31,16 @@

_SUBFOLDERS = {
"hunyuan3d-dit-v2-mv-turbo": "hunyuan3d-dit-v2-mv-turbo",
    "hunyuan3d-dit-v2-mv-fast":  "hunyuan3d-dit-v2-mv-fast",
"hunyuan3d-dit-v2-mv":       "hunyuan3d-dit-v2-mv",
}


class Hunyuan3D2mvGenerator(BaseGenerator):
    MODEL_ID     = "hunyuan3d2mv"
    DISPLAY_NAME = "Hunyuan3D-2mv"
    VRAM_GB      = 8
    MODEL_ID       = "hunyuan3d2mv"
    DISPLAY_NAME   = "Hunyuan3D-2mv"
    VRAM_GB        = 8
    MODEL_VARIANT  = "hunyuan3d-dit-v2-mv-turbo"

# ------------------------------------------------------------------ #
# Lifecycle
@@ -123,13 +131,30 @@ def generate(
):
import torch

        variant        = params.get("model_variant", "hunyuan3d-dit-v2-mv-turbo")
        variant        = params.get("model_variant") or self.MODEL_VARIANT
steps          = int(params.get("num_inference_steps", 30))
octree_res     = int(params.get("octree_resolution", 380))
seed           = int(params.get("seed", 42))
        left_path      = params.get("left_image_path", "")
        back_path      = params.get("back_image_path", "")
        right_path     = params.get("right_image_path", "")

        def _decode_param(key):
            val = params.get(key)
            if val is None:
                return None
            if params.get(key + "_is_b64"):
                return base64.b64decode(val)
            return val

        import base64
        print("[Hunyuan3D2mvGenerator] params keys: %s" % list(params.keys()))
        print("[Hunyuan3D2mvGenerator] left_image present: %s, is_b64: %s" % (
            params.get("left_image") is not None, params.get("left_image_is_b64")))
        print("[Hunyuan3D2mvGenerator] back_image present: %s, is_b64: %s" % (
            params.get("back_image") is not None, params.get("back_image_is_b64")))
        print("[Hunyuan3D2mvGenerator] right_image present: %s, is_b64: %s" % (
            params.get("right_image") is not None, params.get("right_image_is_b64")))
        left_bytes     = _decode_param("left_image")
        back_bytes     = _decode_param("back_image")
        right_bytes    = _decode_param("right_image")

# -- Background removal & preprocessing ---------------------------
self._report(progress_cb, 5, "Preprocessing front view...")
@@ -138,21 +163,24 @@ def generate(

image_dict = {"front": front_image}

        if left_path and os.path.isfile(left_path):
        if left_bytes:
self._report(progress_cb, 10, "Preprocessing left view...")
            image_dict["left"] = self._preprocess_path(left_path)
            image_dict["left"] = self._preprocess_bytes(left_bytes)
self._check_cancelled(cancel_event)

        if back_path and os.path.isfile(back_path):
        if back_bytes:
self._report(progress_cb, 14, "Preprocessing back view...")
            image_dict["back"] = self._preprocess_path(back_path)
            image_dict["back"] = self._preprocess_bytes(back_bytes)
self._check_cancelled(cancel_event)

        if right_path and os.path.isfile(right_path):
        if right_bytes:
self._report(progress_cb, 18, "Preprocessing right view...")
            image_dict["right"] = self._preprocess_path(right_path)
            image_dict["right"] = self._preprocess_bytes(right_bytes)
self._check_cancelled(cancel_event)

        print("[Hunyuan3D2mvGenerator] image_dict keys: %s" % list(image_dict.keys()))
        for k, v in image_dict.items():
            print("[Hunyuan3D2mvGenerator] image_dict[%s] type=%s" % (k, type(v).__name__))
# -- Load variant -------------------------------------------------
self._report(progress_cb, 22, "Loading model variant...")
self._load_variant(variant)
@@ -169,6 +197,19 @@ def generate(
)
t.start()

        # -- Debug: patch prepare_image to log cond_inputs ------------------
        _orig_prepare = self._pipeline.prepare_image
        def _debug_prepare(image):
            result = _orig_prepare(image)
            print("[Hunyuan3D2mvGenerator] prepare_image output keys: %s" % list(result.keys()))
            for k, v in result.items():
                if hasattr(v, 'shape'):
                    print("[Hunyuan3D2mvGenerator] cond_inputs[%s] shape=%s" % (k, list(v.shape)))
                else:
                    print("[Hunyuan3D2mvGenerator] cond_inputs[%s] value=%s" % (k, v))
            return result
        self._pipeline.prepare_image = _debug_prepare

try:
generator = torch.Generator(device=self._device).manual_seed(seed)
with torch.no_grad():
@@ -265,24 +306,24 @@ def params_schema(cls):
"tooltip": "Octree resolution. Higher = more detail but more VRAM.",
},
{
                "id":      "left_image_path",
                "id":      "left_image",
"label":   "Left View Image (optional)",
                "type":    "image_path",
                "default": "",
                "type":    "image",
                "default": None,
"tooltip": "Optional left-side view image.",
},
{
                "id":      "back_image_path",
                "id":      "back_image",
"label":   "Back View Image (optional)",
                "type":    "image_path",
                "default": "",
                "type":    "image",
                "default": None,
"tooltip": "Optional back view image.",
},
{
                "id":      "right_image_path",
                "id":      "right_image",
"label":   "Right View Image (optional)",
                "type":    "image_path",
                "default": "",
                "type":    "image",
                "default": None,
"tooltip": "Optional right-side view image.",
},
{
