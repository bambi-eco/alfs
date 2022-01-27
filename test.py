"""
Example headless EGL rendering on linux in a VM
"""
import numpy as np
from PIL import Image
import moderngl
import alfr.renderer as lfr
from alfr.camera import Camera
from alfr.shot import Shot
from alfr.renderer import Renderer
from pyrr import Matrix44, Quaternion, Vector3, vector
import cv2


# headless:
renderer = Renderer((1024, 1024))
camera = Camera(1.0)

# perspectives of the light field
shots = [
    Shot(
        r"data\debug_scene\0000.png",
        [0, 0, 0],
        [0, 0, 0, 1],
        shot_fovy_degrees=60.0,
    ),
    Shot(
        r"data\debug_scene\0001.png",
        [0.2, 0, 0],
        [0, 0, 0, 1],
        shot_fovy_degrees=60.0,
    ),
    Shot(
        r"data\debug_scene\0014.png",
        [1.0, -1.0, 1.0],
        [0.13052618503570557, 0.0, 0.0, 0.9914448857307434],
        shot_fovy_degrees=60.0,
    ),
]


for i, shot in enumerate(shots):

    vcam = {
        "mat_projection": (camera.mat_projection),
        "mat_lookat": (camera.mat_lookat),
    }

    img = renderer.project_shot(shot, vcam)

    # Pillow image
    #   image = Image.frombytes("RGBA", (512, 512), fbo.read(components=4))
    #   image = image.transpose(Image.FLIP_TOP_BOTTOM)
    #   image.save(f"test{i}.png", format="png")

    # OpenCV image
    # see https://stackoverflow.com/questions/65056007/numpy-array-to-and-from-moderngl-buffer-open-and-save-with-cv2
    #   raw = fbo.read(components=4, dtype="f1")
    #   buf = np.frombuffer(raw, dtype="uint8").reshape((*fbo.size[1::-1], 4))
    cv2.imwrite(f"test{i}.png", img)
