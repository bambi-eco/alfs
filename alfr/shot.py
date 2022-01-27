import numpy as np
import cv2
import moderngl
import alfr.globals as g
from pyrr import Matrix44, Quaternion, Vector3, vector


class Shot:
    """One perspective of the light field"""

    def __init__(
        self,
        shot_filename,
        shot_position,
        shot_rotation,
        shot_fovy_degrees=60.0,
    ):
        # global g.ctx
        if g.ctx is None:
            raise RuntimeError("No OpenGL context available!")

        # one perspective of the light field
        # self.texture = window.load_texture_2d(shot_filename)

        if isinstance(shot_filename, str):
            img = self._load_image(shot_filename)
        elif isinstance(shot_filename, np.ndarray):
            img = shot_filename
        else:
            raise Exception("Unknown type for {shot_filename}")
        self.texture = g.ctx.texture(img.shape[1::-1], img.shape[2], img)

        self.pos = np.array(shot_position)
        self.rot = np.array(shot_rotation)  # rotation as quaternion

        self.fovy = shot_fovy_degrees

    def _load_image(self, texture_filename) -> np.ndarray:
        img = cv2.imread(texture_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB, opencv uses BGR
        img = np.flip(img, 0).copy(order="C")  # flip image vertically
        return img

    def use(self, renderer):
        """
        Use this perspective of the light field.
        """
        self.texture.use(0)

        # get uniforms from shader program
        self.shotViewMat = renderer.program["shotViewMatrix"]
        self.shotProjMat = renderer.program["shotProjectionMatrix"]

        self.shotProjMat.write(
            (Matrix44.perspective_projection(self.fovy, 1.0, 0.01, 100.0)).astype("f4")
        )
        self.shotViewMat.write(
            (
                Matrix44.from_quaternion(self.rot)
                * Matrix44.from_translation(-self.pos)
            ).astype("f4")
        )