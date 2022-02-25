import numpy as np
import cv2
import moderngl
from alfr.globals import ContextManager
from alfr.camera import Camera
from pyrr import Matrix44, Matrix33, Quaternion, Vector3, vector
import json
import os
from typing import Union


class Shot(Camera):
    """One perspective of the light field"""

    def __init__(
        self,
        shot_filename: Union[str, np.ndarray],
        shot_position: Vector3,
        shot_rotation: Quaternion,
        shot_fovy_degrees: float = 60.0,
        shot_aspect_ratio: float = 1.0,
        ctx: moderngl.Context = ContextManager.get_default_context(),
    ):
        super().__init__(
            field_of_view_degrees=shot_fovy_degrees,
            ratio=shot_aspect_ratio,
            position=shot_position,
            quaternion=shot_rotation,
        )

        # global g.ctx
        if ctx is None:
            raise RuntimeError("No OpenGL context available!")

        # one perspective of the light field
        # self.texture = window.load_texture_2d(shot_filename)
        self._filename = None
        if isinstance(shot_filename, str):
            img = self._load_image(shot_filename)
            self._filename = shot_filename
        elif isinstance(shot_filename, np.ndarray):
            img = shot_filename
        else:
            raise Exception("Unknown type for {shot_filename}")
        self.texture = ctx.texture(img.shape[1::-1], img.shape[2], img)
        self._img = img  # opencv image

    @property
    def image_file(self):
        return self._filename

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

        # get uniforms from shader program and set them
        renderer.program["m_shot_proj"].write(self.projection_matrix.astype("f4"))
        renderer.program["m_shot_cam"].write(self.view_matrix.astype("f4"))
