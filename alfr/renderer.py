import numpy as np
import cv2
import moderngl
import alfr.globals as g
from alfr.shot import Shot
from typing import Tuple
from pyrr import Matrix44, Quaternion, Vector3, vector


def plane(size):
    """
    Create a plane with the given size.
    """
    u = np.repeat(np.linspace(-size, size, 2), 2)
    v = np.tile([-size, size], 2)
    w = np.ones(4) * -15
    return np.concatenate([np.dstack([u, v, w]), np.dstack([v, u, w])])


class Renderer:
    def __init__(self, resolution: tuple = (512, 512)):
        # g.ctx
        if g.ctx is None:
            g.ctx = moderngl.create_context(standalone=True)
        self._ctx = g.ctx

        # Todo: test on Colab and so forth

        self._program = self._setup_alfr_program(self._ctx)
        self._fbo = self._ctx.simple_framebuffer(resolution, components=4)

        vbo = self._ctx.buffer(plane(15).astype("f4"))
        # Indices are given to specify the order of drawing
        indices = np.array([0, 1, 2, 2, 3, 1], dtype="i4")
        ibo = self._ctx.buffer(indices)
        vao_content = [
            # 3 floats are assigned to the 'in' variable named 'in_vert' in the shader code
            (vbo, "3f", "in_vert")
        ]
        self._vao = self._ctx.vertex_array(self._program, vao_content, ibo)

    def project_shot(
        self, shot: Shot, vcam: dict, focus=None, resolution=None
    ) -> np.ndarray:
        """
        Project the given camera into a given shot.
        """

        if resolution is not None and resolution != self._fbo.size:
            self.fbo = self._ctx.simple_framebuffer(resolution, components=4)
        self.fbo.use()

        self._ctx.clear(1.0, 1.0, 1.0)
        self._ctx.enable(moderngl.DEPTH_TEST)

        modelMat = self._program["modelMatrix"]
        viewMat = self._program["viewMatrix"]
        projMat = self._program["projectionMatrix"]

        projMat.write((vcam["mat_projection"]).astype("f4"))
        viewMat.write((vcam["mat_lookat"]).astype("f4"))
        modelMat.write((Matrix44.identity()).astype("f4"))

        shot.use(self)
        self._vao.render(moderngl.TRIANGLES)

        # opencv image
        # see https://stackoverflow.com/questions/65056007/numpy-array-to-and-from-moderngl-buffer-open-and-save-with-cv2
        raw = self.fbo.read(components=4, dtype="f1")
        img = np.frombuffer(raw, dtype="uint8").reshape((*self.fbo.size[1::-1], 4))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB, opencv uses BGR
        img = np.flip(img, 0).copy(order="C")  # flip image vertically

        # flip red and blue channels; cvtColor removes alpha channel so do it manually!
        tmp = img[:, :, 0].copy()
        img[:, :, 0] = img[:, :, 2].copy()
        img[:, :, 2] = tmp
        return img

    @property
    def fbo(self):
        return self._fbo

    @fbo.setter
    def fbo(self, fbo: moderngl.Framebuffer):
        self._fbo = fbo

    @property
    def program(self):
        return self._program

    def _setup_alfr_program(self, ctx: moderngl.Context) -> moderngl.Program:
        return ctx.program(
            vertex_shader="""
                    #version 330

                    // model view projection matrix of the model (virtual camera)
                    uniform mat4 modelMatrix;
                    uniform mat4 viewMatrix;
                    uniform mat4 projectionMatrix;

                    // view and camera/projection matrix for one shot:
                    uniform mat4 shotViewMatrix;
                    uniform mat4 shotProjectionMatrix;

                    in vec3 in_vert;
                    out vec4 wpos;
                    out vec4 shotUV;

                    void main() {
                        wpos = modelMatrix * vec4(in_vert, 1.0);
                        gl_Position = projectionMatrix * viewMatrix * wpos;

                        shotUV = shotProjectionMatrix * shotViewMatrix * wpos;
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
                            color = vec4(texture(shotTexture, uv.xy).rgb, 1.0);
                        }
                    }
                """,
        )
