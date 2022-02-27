from .thirdparty.read_write_model import (
    read_model,
)  # from https://github.com/colmap/colmap
import moderngl
from alfr.globals import ContextManager
from alfr.camera import Camera
from alfr.shot import Shot
from pyrr import Matrix44, Matrix33, Quaternion, Vector3, vector
from typing import List
import json
import os
import numpy as np


def get_from_dict(d: dict, keys: list):
    for key in keys:
        if key in d.keys():
            return d[key]
    return None


def get_file_pos_rot(images: dict):
    file = get_from_dict(images, ["imagefile", "file", "image"])
    pos = get_from_dict(images, ["location", "pos", "loc"])
    rot = get_from_dict(images, ["rotation", "rot", "quaternion"])
    fov = get_from_dict(images, ["fovy", "fov", "fieldofview"])

    if file is None or pos is None or rot is None:
        raise Exception("Not all keys found in images dict!")

    return file, pos, rot, fov


def export_shots_to_json(
    shots: List[Shot],
    json_file: str,
):
    """
    Exports shots to a json file.
    """
    data = {"images": []}
    for shot in shots:
        data["images"].append(
            {
                "imagefile": os.path.basename(shot.image_file),
                "location": shot.position.tolist(),
                "rotation": shot.rotation.tolist(),
                "fovy": shot.fov_degree,
            }
        )

    with open(json_file, "w") as f:
        json.dump(data, f)


def load_shots_from_json(
    json_file: str,
    fovy: float = 60.0,
    ctx: moderngl.Context = ContextManager.get_default_context(),
):
    """
    Loads shots from a json file.
    """
    shots = []
    with open(json_file, "r") as f:
        data = json.load(f)

        json_dir = os.path.dirname(os.path.realpath(f.name))

        if "images" in data.keys():
            for image in data["images"]:
                file, pos, rot, fov = get_file_pos_rot(image)
                shot = Shot(
                    os.path.join(json_dir, file),
                    Vector3(pos),
                    Quaternion(rot),  # format x,y,z,w
                    fov if fov is not None else fovy,
                    shot_aspect_ratio=1.0,
                    ctx=ctx,
                )
                shots.append(shot)

    return shots


def load_shots_from_legacy_json(
    json_file: str,
    fovy: float = 60.0,
    ctx: moderngl.Context = ContextManager.get_default_context(),
):
    """
    Loads shots from a legacy json file.
    """
    shots = []
    with open(json_file, "r") as f:
        data = json.load(f)

        json_dir = os.path.dirname(os.path.realpath(f.name))

        # m = Matrix33([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        flip_yz_matrix = Matrix33([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        if "images" in data.keys():
            for image in data["images"]:
                file = get_from_dict(image, ["imagefile", "file", "image"])
                M_3x4: list = get_from_dict(image, ["M3x4"])
                weird_R = Matrix33([M_3x4[0][:3], M_3x4[1][:3], M_3x4[2][:3]])
                weird_t = Vector3(np.asarray(M_3x4)[:, 3])

                R: Matrix33 = flip_yz_matrix @ weird_R.transpose() @ flip_yz_matrix
                rot = R.quaternion

                pos = -R @ weird_t @ flip_yz_matrix

                # for some reason we need to modify the quaternion here.
                # Colmap has a weird format???
                qt = rot
                _q = Quaternion(np.hstack((qt.w, qt.z, -qt.y, qt.x)))
                # Todo: verify why we need this!!

                shot = Shot(
                    os.path.join(json_dir, file),
                    Vector3(pos),
                    _q,
                    fovy,
                    shot_aspect_ratio=1.0,
                    ctx=ctx,
                )
                shots.append(shot)

    return shots


# Todo!!
def load_shots_from_colmap(
    model_folder: str,
    image_folder: str,
    fovy: float = None,
    ctx: moderngl.Context = ContextManager.get_default_context(),
):
    """
    Loads shots from a colmap.
    """

    cameras, images, points3D = read_model(model_folder)  # read the colmap model

    # Todo: finish this!

    shots = []
    for img in images.values():
        # look into https://github.com/colmap/colmap/blob/dev/scripts/python/visualize_model.py
        # to see how to get the camera parameters

        # rotation
        q = Quaternion((img.qvec[1], img.qvec[2], img.qvec[3], img.qvec[0]))
        # pyrr.Quaternion format is x,y,z,w while colmap uses w,x,y,z!!!
        R = Matrix33.from_quaternion(q)

        # translation
        t = Vector3(img.tvec)

        # invert
        t = -R.T @ t
        _R = Matrix33(R.T)

        # intrinsics
        cam = cameras[img.camera_id]

        if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = cam.params[0]
            cx = cam.params[1]
            cy = cam.params[2]
        elif cam.model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE"):
            fx = cam.params[0]
            fy = cam.params[1]
            cx = cam.params[2]
            cy = cam.params[3]
        else:
            raise Exception("Camera model not supported")

        # intrinsics
        K = np.identity(3)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy

        # field of view in degrees
        cam_fovy = np.degrees(
            2 * np.arctan(cam.height / (2 * fy))
        )  # Todo: check if width or height is correct here!
        # print("fovy:", cam_fovy)

        # for some reason we need to modify the quaternion here.
        # Colmap has a weird format???
        qt = _R.quaternion
        _q = Quaternion(np.hstack((qt.w, qt.z, -qt.y, qt.x)))
        # Todo: verify why we need this!!

        shot = Shot(
            os.path.join(image_folder, img.name),
            Vector3(t),
            _q,
            fovy if fovy is not None else cam_fovy,
            shot_aspect_ratio=cam.width / cam.height,
            ctx=ctx,
        )
        shots.append(shot)

    return shots
