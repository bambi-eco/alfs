"""
    Added a simple camera class to an existing example.
    The camera class is built using following tutorials:
       https://learnopengl.com/Getting-started/Camera
       http://in2gpu.com/2016/03/14/opengl-fps-camera-quaternion/

    Controls:
        Move:
            Forward - W
            Backwards - S

        Strafe:
            Up - up arrow
            Down - down arrow
            Left - A
            Right - D

        Rotate:
            Left - Q
            Right - E

        Zoom:
            In - X
            Out - Z

    adopted by: Alex Zakrividoroga
"""
import os
import numpy as np
from pyrr import Matrix44, Quaternion, Vector3, vector

import moderngl
import moderngl_window as mglw


class Example(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "ModernGL Example"
    window_size = (1280, 720)
    aspect_ratio = 16 / 9
    resizable = True

    resource_dir = os.path.normpath(os.path.join("."))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def run(cls):
        mglw.run_window_config(cls)


class Camera:
    def __init__(self, ratio):
        self._zoom_step = 0.1
        self._move_vertically = 0.1
        self._move_horizontally = 0.1
        self._rotate_horizontally = 0.1
        self._rotate_vertically = 0.1

        self._field_of_view_degrees = 60.0
        self._z_near = 0.1
        self._z_far = 100
        self._ratio = ratio
        self.build_projection()

        self._camera_position = Vector3([0.0, 0.0, 0.0])
        self._camera_front = Vector3([0.0, 0.0, -1.0])
        self._camera_up = Vector3([0.0, 1.0, 0.0])
        self._cameras_target = self._camera_position + self._camera_front
        self.build_look_at()

    def zoom_in(self):
        self._field_of_view_degrees = self._field_of_view_degrees - self._zoom_step
        self.build_projection()

    def zoom_out(self):
        self._field_of_view_degrees = self._field_of_view_degrees + self._zoom_step
        self.build_projection()

    def move_forward(self):
        self._camera_position = (
            self._camera_position + self._camera_front * self._move_horizontally
        )
        self.build_look_at()

    def move_backwards(self):
        self._camera_position = (
            self._camera_position - self._camera_front * self._move_horizontally
        )
        self.build_look_at()

    def strafe_left(self):
        self._camera_position = (
            self._camera_position
            - vector.normalize(self._camera_front ^ self._camera_up)
            * self._move_horizontally
        )
        self.build_look_at()

    def strafe_right(self):
        self._camera_position = (
            self._camera_position
            + vector.normalize(self._camera_front ^ self._camera_up)
            * self._move_horizontally
        )
        self.build_look_at()

    def strafe_up(self):
        self._camera_position = (
            self._camera_position + self._camera_up * self._move_vertically
        )
        self.build_look_at()

    def strafe_down(self):
        self._camera_position = (
            self._camera_position - self._camera_up * self._move_vertically
        )
        self.build_look_at()

    def rotate_left(self):
        rotation = Quaternion.from_y_rotation(
            2 * float(self._rotate_horizontally) * np.pi / 180
        )
        self._camera_front = rotation * self._camera_front
        self.build_look_at()

    def rotate_right(self):
        rotation = Quaternion.from_y_rotation(
            -2 * float(self._rotate_horizontally) * np.pi / 180
        )
        self._camera_front = rotation * self._camera_front
        self.build_look_at()

    def build_look_at(self):
        self._cameras_target = self._camera_position + self._camera_front
        self.mat_lookat = Matrix44.look_at(
            self._camera_position, self._cameras_target, self._camera_up
        )

    def build_projection(self):
        self.mat_projection = Matrix44.perspective_projection(
            self._field_of_view_degrees, self._ratio, self._z_near, self._z_far
        )


def grid(size, steps):
    u = np.repeat(np.linspace(-size, size, steps), 2)
    v = np.tile([-size, size], steps)
    w = np.zeros(steps * 2)
    return np.concatenate([np.dstack([u, v, w]), np.dstack([v, u, w])])


def plane(size):
    """
    Create a plane with the given size.
    """
    u = np.repeat(np.linspace(-size, size, 2), 2)
    v = np.tile([-size, size], 2)
    w = np.ones(4) * -4
    return np.concatenate([np.dstack([u, v, w]), np.dstack([v, u, w])])


class Shot:
    """One perspective of the light field"""

    def __init__(
        self,
        shot_filename,
        shot_position,
        shot_rotation,
        program: moderngl.Program,
        window: mglw.WindowConfig,
        shot_fovy_degrees=60.0,
    ):

        # one perspective of the light field
        self.texture = window.load_texture_2d(shot_filename)
        self.pos = np.array(shot_position)
        self.rot = np.array(shot_rotation)  # rotation as quaternion

        self.fovy = shot_fovy_degrees

        # get uniforms from shader program
        self.shotViewMat = program["shotViewMatrix"]
        self.shotProjMat = program["shotProjectionMatrix"]

    def use(self):
        """
        Use this perspective of the light field.
        """
        self.texture.use(0)

        self.shotProjMat.write(
            (Matrix44.perspective_projection(self.fovy, 1.0, 0.01, 100.0)).astype("f4")
        )
        self.shotViewMat.write(
            (
                Matrix44.from_quaternion(self.rot)
                * Matrix44.from_translation(-self.pos)
            ).astype("f4")
        )


class PerspectiveProjection(Example):
    gl_version = (3, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330

                uniform mat4 modelMatrix;
                uniform mat4 viewMatrix;
                uniform mat4 projectionMatrix;

                in vec3 in_vert;
                out vec4 wpos;

                void main() {
                    wpos = modelMatrix * vec4(in_vert, 1.0);
                    gl_Position = projectionMatrix * viewMatrix * wpos;
                }
            """,
            fragment_shader="""
                #version 330

                uniform mat4 shotViewMatrix;
                uniform mat4 shotProjectionMatrix;
                uniform sampler2D shotTexture;

                in vec4 wpos;
                out vec4 color;

                void main() {
                    vec4 uv = shotProjectionMatrix * shotViewMatrix * wpos;
                    uv = vec4(uv.xyz / uv.w / 2.0 + .5, 1.0); // perspective division and converstion to [0,1]

                    if(uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
                        discard;
                        color = vec4(0.0, 0.0, 0.0, 0.0);
                    } else {
                        color = vec4(texture(shotTexture, uv.xy).rgb, 1.0);
                    }
                }
            """,
        )

        self.camera = Camera(self.aspect_ratio)
        self.modelMat = self.prog["modelMatrix"]
        self.viewMat = self.prog["viewMatrix"]
        self.projMat = self.prog["projectionMatrix"]
        self.shotViewMat = self.prog["shotViewMatrix"]
        self.shotProjMat = self.prog["shotProjectionMatrix"]
        # self.vbo = self.ctx.buffer(grid(15, 10).astype("f4"))
        self.vbo = self.ctx.buffer(plane(15).astype("f4"))
        # Indices are given to specify the order of drawing
        indices = np.array([0, 1, 2, 2, 3, 1], dtype="i4")
        self.ibo = self.ctx.buffer(indices)
        vao_content = [
            # 3 floats are assigned to the 'in' variable named 'in_vert' in the shader code
            (self.vbo, "3f", "in_vert")
        ]
        self.vao = self.ctx.vertex_array(self.prog, vao_content, self.ibo)

        # perspectives of the light field
        self.shots = [
            Shot(
                r"data\debug_scene\0000.png",
                [0, 0, 0],
                [0, 0, 0, 1],
                self.prog,
                self,
                shot_fovy_degrees=60.0,
            ),
            Shot(
                r"data\debug_scene\0001.png",
                [0.2, 0, 0],
                [0, 0, 0, 1],
                self.prog,
                self,
                shot_fovy_degrees=60.0,
            ),
            Shot(
                r"data\debug_scene\0014.png",
                [1.0, -1.0, 1.0],
                [0.13052618503570557, 0.0, 0.0, 0.9914448857307434],
                self.prog,
                self,
                shot_fovy_degrees=60.0,
            ),
        ]

        self.states = {
            self.wnd.keys.W: False,  # forward
            self.wnd.keys.S: False,  # backwards
            self.wnd.keys.UP: False,  # strafe Up
            self.wnd.keys.DOWN: False,  # strafe Down
            self.wnd.keys.A: False,  # strafe left
            self.wnd.keys.D: False,  # strafe right
            self.wnd.keys.Q: False,  # rotate left
            self.wnd.keys.E: False,  # rotare right
            self.wnd.keys.Z: False,  # zoom in
            self.wnd.keys.X: False,  # zoom out
        }

    def move_camera(self):
        if self.states.get(self.wnd.keys.W):
            self.camera.move_forward()

        if self.states.get(self.wnd.keys.S):
            self.camera.move_backwards()

        if self.states.get(self.wnd.keys.UP):
            self.camera.strafe_up()

        if self.states.get(self.wnd.keys.DOWN):
            self.camera.strafe_down()

        if self.states.get(self.wnd.keys.A):
            self.camera.strafe_left()

        if self.states.get(self.wnd.keys.D):
            self.camera.strafe_right()

        if self.states.get(self.wnd.keys.Q):
            self.camera.rotate_left()

        if self.states.get(self.wnd.keys.E):
            self.camera.rotate_right()

        if self.states.get(self.wnd.keys.Z):
            self.camera.zoom_in()

        if self.states.get(self.wnd.keys.X):
            self.camera.zoom_out()

    def key_event(self, key, action, modifiers):
        if key not in self.states:
            print(key, action)
            return

        if action == self.wnd.keys.ACTION_PRESS:
            self.states[key] = True
        else:
            self.states[key] = False

    def render(self, time, frame_time):
        self.move_camera()

        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.projMat.write((self.camera.mat_projection).astype("f4"))
        self.viewMat.write((self.camera.mat_lookat).astype("f4"))
        self.modelMat.write((Matrix44.identity()).astype("f4"))

        self.vao.render(moderngl.TRIANGLES)

        self.shots[round(time * 10) % len(self.shots)].use()
        self.vao.render()


if __name__ == "__main__":
    PerspectiveProjection.run()
