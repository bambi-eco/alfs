"""
    Added a simple camera class to an existing example.
    The camera class is taken from the moderngl examples:
    https://github.com/moderngl/moderngl/blob/master/examples/simple_camera.py
    (Alex Zakrividoroga)
"""
import numpy as np
from pyrr import Matrix44, Quaternion, Vector3, vector


class Camera:
    def __init__(
        self,
        field_of_view_degrees: float = 60,
        ratio: float = 1.0,
        z_near: float = 0.1,
        z_far: float = 10000,
        position=Vector3([0, 0, 0]),
        quaternion=Quaternion([0,0,0,1]),
        camera_front: Vector3 = None,
        camera_up: Vector3 = None,
    ):
        self._field_of_view_degrees = field_of_view_degrees
        self._z_near = z_near
        self._z_far = z_far
        self._ratio = ratio

        self._camera_position = Vector3(position)
        self._rotation = Quaternion(quaternion)
        if camera_front is None or camera_up is None:
            self._camera_front = self.rotation * Vector3([0, 0, -1])
            print(f"camera_front: {self._camera_front}")
            self._camera_up = self.rotation * Vector3([0, 1, 0])
        else:
            # use front and up vector to create a view matrix!
            self._camera_front = camera_front
            self._camera_up = camera_up
            self._rotation = Quaternion.from_matrix(self._build_look_at())

        self._cameras_target = self._camera_position + self._camera_front

    @property
    def position(self) -> Vector3:
        return self._camera_position

    @property
    def rotation(self) -> Quaternion:
        return self._rotation

    @property
    def projection_matrix(self) -> Matrix44:
        return Matrix44.perspective_projection(
            self._field_of_view_degrees, self._ratio, self._z_near, self._z_far
        )

    @property
    def view_matrix(self) -> Matrix44:
        return Matrix44.from_quaternion(self.rotation) * Matrix44.from_translation(
            -self.position
        )

    def _build_look_at(self):
        self._cameras_target = self._camera_position + self._camera_front
        return Matrix44.look_at(
            self._camera_position, self._cameras_target, self._camera_up
        )


class ControllableCamera(Camera):
    def __init__(self, ratio: float = 1.0):
        super().__init__(
            position=Vector3([0.0, 0.0, 0.0]),
            camera_front=Vector3([0.0, 0.0, -1.0]),
            camera_up=Vector3([0.0, 1.0, 0.0]),
        )

        self._zoom_step = 0.1
        self._move_vertically = 0.1
        self._move_horizontally = 0.1
        self._rotate_horizontally = 0.1
        self._rotate_vertically = 0.1

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
