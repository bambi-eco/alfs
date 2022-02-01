"""
Example headless EGL rendering on linux in a VM
"""
import numpy as np
from PIL import Image
import moderngl
import alfr
from pyrr import Matrix44, Quaternion, Vector3, vector
import cv2
import time


# headless:
renderer = alfr.Renderer((512, 512))
camera = alfr.Camera(position=[0,1,0],camera_front=[0,0,-1],camera_up=[0,-1,0])

# load perspectives of the light field
shots = alfr.load_shots_from_json(r"data\debug_scene\blender_poses.json", fovy=60.0)

vcam = camera

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
