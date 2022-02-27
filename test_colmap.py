"""
Example headless EGL rendering on linux in a VM
"""
import numpy as np
from PIL import Image
import moderngl
import alfr
from pyrr import Matrix44, Matrix33, Quaternion, Vector3, vector
import cv2
import time


# headless:
renderer = alfr.Renderer((512, 512))
camera = alfr.Camera(position=[0, 1, 0], camera_front=[0, 0, -1], camera_up=[0, -1, 0])


# Todo!!!

# DEBUGGING with colmap_debug scene!!!
"""
# Matrix from MeshLab
m = Matrix33(
    [
        [0.999838, 0.01087, -0.0143608],
        [0.0108437, -0.999939, -0.00190539],
        [-0.0143807, 0.00174936, -0.999895],
    ]
)
print("test Matrix ", m)
print("its quaternion ", m.quaternion)
# Somehow the quaternions are turned/flipped in a weird way!!!

shots = alfr.load_shots_from_colmap(
    r".\data\debug_reconstruction\COLMAP",
    r".\data\debug_reconstruction\images",
    fovy=60.0,
)

for i, shot in enumerate(shots):
    print(f"camera {i} pos: {shot.position}")
    print(f"camera {i} quaternion: {shot.rotation}")
    print(f"camera {i} R: {Matrix33(shot.rotation)}")
"""


# load perspectives of the light field ** F5 **
shots = alfr.load_shots_from_colmap(
    r"data/colmap_scene/F5/poses/COLMAP",
    r"data\colmap_scene\F5\RGB_renamed",
    fovy=50.81543,
)
alfr.export_shots_to_json(shots, r"data\colmap_scene\F5\RGB_renamed\new_poses.json")


vcam = shots[0]

for i, shot in enumerate(shots):

    start = time.time()
    img = renderer.project_shot(shot, vcam)
    end = time.time()
    print(f"Projection of {i}: {(end-start)*1000} milli seconds")
    # Pillow image
    #   image = Image.frombytes("RGBA", (512, 512), fbo.read(components=4))
    #   image = image.transpose(Image.FLIP_TOP_BOTTOM)
    #   image.save(f"test{i}.png", format="png")

    # OpenCV image
    # see https://stackoverflow.com/questions/65056007/numpy-array-to-and-from-moderngl-buffer-open-and-save-with-cv2
    #   raw = fbo.read(components=4, dtype="f1")
    #   buf = np.frombuffer(raw, dtype="uint8").reshape((*fbo.size[1::-1], 4))
    cv2.imwrite(f"test{i}.png", img)

start = time.time()
imgs = renderer.project_multiple_shots(shots, vcam)
end = time.time()
print(f"Projection of {len(imgs)} images: {(end-start)*1000} milli seconds")
for i, img in enumerate(imgs):
    cv2.imwrite(f"test_multiple_projections_{i}.png", img)


# integral ----
start = time.time()
integral = np.stack([renderer.project_shot(shot, vcam) for shot in shots], axis=-1).sum(
    axis=-1
)
alpha = integral[:, :, -1] / 255.0
# print(alpha)
integral = np.divide(integral, alpha[:, :, np.newaxis])
end = time.time()
print(f"Integral computation, {len(shots)} shots: {(end-start)*1000} milli seconds")
# print(integral)
cv2.imwrite(f"integral.png", integral[:, :, :3])


start = time.time()
integral = renderer.integrate(shots, vcam)
end = time.time()
print(
    f"Integral (renderer) computation, {len(shots)} shots: {(end-start)*1000} milli seconds"
)
# print(integral)
cv2.imwrite(f"integral_renderer.png", integral[:, :, :3])
