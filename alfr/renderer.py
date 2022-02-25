import numpy as np
import cv2
import moderngl
from alfr.globals import ContextManager
from alfr.shot import Shot
from alfr.camera import Camera
from typing import Tuple
from pyrr import Matrix44, Quaternion, Vector3, vector
from typing import List


def plane(size):
    """
    Create a plane with the given size.
    """
    u = np.repeat(np.linspace(-size, size, 2), 2)
    v = np.tile([-size, size], 2)
    w = np.ones(4) * -9
    return np.concatenate([np.dstack([u, v, w]), np.dstack([v, u, w])])


class Renderer:
    def __init__(
        self,
        resolution: tuple = (512, 512),
        ctx: moderngl.Context = ContextManager.get_default_context(),
    ):

        self._ctx = ctx
        self._program = self._setup_alfr_program(self._ctx)
        self._fbo = self._ctx.simple_framebuffer(resolution, components=4)

        vbo = self._ctx.buffer(plane(100).astype("f4"))
        # Indices are given to specify the order of drawing
        indices = np.array([0, 1, 2, 2, 3, 1], dtype="i4")
        ibo = self._ctx.buffer(indices)
        vao_content = [
            # 3 floats are assigned to the 'in' variable named 'in_vert' in the shader code
            (vbo, "3f", "in_position")
        ]
        self._vao = self._ctx.vertex_array(self._program, vao_content, ibo)

    def _prepare_projection(self, vcam: Camera, focus=None, resolution: tuple = None):
        """Prepare the renderer for projection a shot.

        Activate the framebuffer, clear it and set the matrices for the shader program.

        Args:
            vcam (Camera): the virtual camera
            focus (float): the focus object
            resolution (tuple): the resolution of the image

        """

        if resolution is not None and resolution != self._fbo.size:
            self.fbo = self._ctx.simple_framebuffer(resolution, components=4)
        self.fbo.use()

        self._ctx.clear(0.0, 0.0, 0.0)
        self._ctx.enable(moderngl.DEPTH_TEST)

        modelMat = self._program["m_model"]
        viewMat = self._program["m_cam"]
        projMat = self._program["m_proj"]

        projMat.write(vcam.projection_matrix.astype("f4"))
        viewMat.write(vcam.view_matrix.astype("f4"))
        modelMat.write((Matrix44.identity()).astype("f4"))  # Todo!

    def _img_from_fbo(self) -> np.ndarray:
        """Get the image from the framebuffer.

        Returns:
            np.ndarray: the image
        """
        # opencv image
        # see https://stackoverflow.com/questions/65056007/numpy-array-to-and-from-moderngl-buffer-open-and-save-with-cv2
        raw = self.fbo.read(components=4, dtype="f1")
        return np.frombuffer(raw, dtype="uint8").reshape((*self.fbo.size[1::-1], 4))

    def _postpro_img(self, img: np.ndarray) -> np.ndarray:
        """Postprocess an image such that it is compatible with opencv

        Args:
            img (np.ndarray): the image to process

        Returns:
            np.ndarray: processed image
        """
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB, opencv uses BGR
        img = np.flip(img, 0).copy(order="C")  # flip image vertically
        # flip red and blue channels; cvtColor removes alpha channel so do it manually!
        tmp = img[:, :, 0].copy()
        img[:, :, 0] = img[:, :, 2].copy()
        img[:, :, 2] = tmp
        return img

    def project_shot(
        self, shot: Shot, vcam: Camera, focus=None, resolution=None
    ) -> np.ndarray:
        """Project the given camera into a given shot.

        Args:
            shot (Shot): the shot to project
            vcam (Camera): the virtual camera
            focus (float): the focus object
            resolution (tuple): the resolution of the image

        Returns:
            np.ndarray: the projected image
        """

        self._prepare_projection(vcam, focus, resolution)

        self._ctx.clear(0.0, 0.0, 0.0)
        shot.use(self)
        self._vao.render(moderngl.TRIANGLES)

        img = self._img_from_fbo()
        return self._postpro_img(img)

    def project_multiple_shots(
        self,
        shots: List[Shot],
        vcam: Camera,
        focus=None,
        resolution=None,
        postprocess=True,
    ) -> List[np.ndarray]:
        """Project multiple shots into images.

        Args:
            shots (List[Shot]): the shots to project
            vcam (Camera): the virtual camera
            focus (float): the focus object
            resolution (tuple): the resolution of the image
            postprocess (bool): whether to postprocess the image

        Returns:
            List[np.ndarray]: the projected images
        """
        projections = []
        self._prepare_projection(vcam, focus, resolution)

        for shot in shots:
            self._ctx.clear(0.0, 0.0, 0.0)
            shot.use(self)
            self._vao.render(moderngl.TRIANGLES)

            img = self._img_from_fbo()
            projections.append(self._postpro_img(img) if postprocess else img)

        return projections

    def integrate(
        self, shots: List[Shot], vcam: Camera, focus=None, resolution: tuple = None
    ) -> np.ndarray:
        """Integrate multiple shots into a single image.

        Args:
            shots (List[Shot]): the shots to integrate
            vcam (Camera): the virtual camera
            focus (float): the focus object
            resolution (tuple): the resolution of the image

        Returns:
            np.ndarray: the integrated image
        """
        projections = self.project_multiple_shots(
            shots, vcam, focus, resolution, postprocess=False
        )

        integral = np.stack(projections, axis=-1).sum(axis=-1)
        integral = self._postpro_img(integral)  # postprocess only once!
        alpha = integral[:, :, -1] / 255.0
        integral = np.divide(integral, alpha[:, :, np.newaxis])
        return integral

    @property
    def fbo(self):
        """Get or Set the internal framebuffer used by the renderer."""
        return self._fbo

    @fbo.setter
    def fbo(self, fbo: moderngl.Framebuffer):
        self._fbo = fbo

    @property
    def program(self):
        """The internal shader program used by the renderer."""
        return self._program

    @staticmethod
    def _setup_alfr_program(ctx: moderngl.Context) -> moderngl.Program:
        """Setup the shader program to be used by the renderer."""
        return ctx.program(
            vertex_shader="""
                    #version 330

                    // model view projection matrices of the focus surface (virtual camera)
                    uniform mat4 m_proj;
                    uniform mat4 m_model;
                    uniform mat4 m_cam;

                    // view and camera/projection matrix for one shot:
                    uniform mat4 m_shot_cam;
                    uniform mat4 m_shot_proj;

                    in vec3 in_position;
                    out vec4 wpos;
                    out vec4 shotUV;

                    void main() {
                        wpos = m_model * vec4(in_position, 1.0);
                        gl_Position = m_proj * m_cam * wpos;

                        shotUV = m_shot_proj * m_shot_cam * wpos;
                    }
                """,
            fragment_shader="""
                    #version 330


                    uniform sampler2D shotTexture;

                    in vec4 wpos;
                    in vec4 shotUV;
                    out vec4 color;

                    void main() {
                        vec4 uv = shotUV;
                        uv = vec4(uv.xyz / uv.w / 2.0 + .5, 1.0); // perspective division and conversion to [0,1] from NDC

                        if(uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
                            discard; // throw away the fragment 
                            color = vec4(0.0, 0.0, 0.0, 0.0);
                        } else {
                            // DEBUG: color = vec4(1.0, 1.0, 0.0, 1.0);
                            color = vec4(texture(shotTexture, uv.xy).rgb, 1.0);
                        }
                    }
                """,
        )
