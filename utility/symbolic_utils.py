import sympy as sp
from typing import List, Optional


def inverse_transform(T: sp.Matrix) -> sp.Matrix:
    """
    Compute the inverse of a 4x4 homogeneous transformation matrix
    :param T: the input 4x4 matrix
    :return:
    """
    rot = T[0:3, 0:3].T
    translation = -rot * T[0:3, 3]
    invT = rot.row_join(translation)
    invT = invT.col_join(sp.Matrix([0, 0, 0, 1]).T)
    return invT


def inverse_auxiliary_transform(transform: sp.Matrix, parameter_values=None) -> sp.Matrix:
    """Invert a stored base/tool frame, including rounded rotation entries.

    DH transforms retain the fast rigid inverse above. Auxiliary frames may
    contain rounded constants whose transpose is not the configured inverse.
    Runtime frame parameters must describe nonsingular affine transforms.
    """
    values = parameter_values or {}
    if transform.shape != (4, 4) or any(
            sp.simplify((transform[3, col]-(1 if col == 3 else 0)).subs(values)).is_zero is not True
            for col in range(4)):
        raise ValueError('Base/tool transforms must have homogeneous bottom row [0, 0, 0, 1]')
    rotation = transform[:3, :3]
    known = rotation.subs(values)
    if not known.free_symbols and any(value.is_real is not True or value.is_finite is not True for value in known):
        raise ValueError('Base/tool rotation entries must be finite real numbers')
    if known.det().is_zero is True:
        raise ValueError('Base/tool transforms must be nonsingular')
    residual = rotation.T*rotation-sp.eye(3)
    if all(sp.trigsimp(value).is_zero is True for value in residual):
        return inverse_transform(transform)
    inverse = rotation.inv().applyfunc(sp.trigsimp)
    return inverse.row_join(-inverse*transform[:3, 3]).col_join(sp.Matrix([[0, 0, 0, 1]]))


def inverse_rotation(rotation_mat: sp.Matrix) -> sp.Matrix:
    """
    Compute the inverse of a rotation matrix
    :param rotation_mat:
    :return:
    """
    return rotation_mat.T


def rotation_z(theta: sp.Symbol) -> sp.Matrix:
    """
    4x4 rotation matrix in z
    :param theta: rotation angle
    :return: 4x4 rotation matrix
    """
    t = sp.Matrix([
        [sp.cos(theta), -sp.sin(theta), 0, 0],
        [sp.sin(theta),  sp.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return t


def rotation_x(theta: sp.Symbol) -> sp.Matrix:
    """
    4x4 rotation matrix in x
    :param theta: rotation angle
    :return: 4x4 rotation matrix
    """
    t = sp.Matrix([
        [1, 0, 0, 0],
        [0, sp.cos(theta), -sp.sin(theta), 0],
        [0, sp.sin(theta),  sp.cos(theta), 0],
        [0, 0, 0, 1]
    ])
    return t


def translation_z(d: sp.Symbol) -> sp.Matrix:
    """
    4x4 translation matrix in z
    :param d: translation distance
    :return: 4x4 translation matrix
    """
    t = sp.Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, d],
        [0, 0, 0, 1]
    ])
    return t


def translation_x(a: sp.Symbol) -> sp.Matrix:
    """
    4x4 translation matrix in x
    :param a: translation distance
    :return: 4x4 translation matrix
    """
    t = sp.Matrix([
        [1, 0, 0, a],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return t


def is_identity_matrix(matrix: sp.Matrix) -> bool:
    """
    Determine wheter a matrix is a identity matrix
    :param matrix:
    :return:
    """
    if matrix.rows != matrix.cols:
        return False
    for i in range(matrix.rows):
        if matrix[i, i] != sp.S.One:
            return False
    for i in range(matrix.rows):
        for j in range(matrix.cols):
            if i == j:
                continue
            if matrix[i, j] != sp.S.Zero:
                return False
    return True


def multiple_list_of_transforms(transform_list: List[Optional[sp.Matrix]]) -> sp.Matrix:
    """
    Given a list of transforms, where None represents identify, multiple them together
    to get the final transform.
    :param transform_list:
    :return:
    """
    result = sp.eye(4)
    for i in range(len(transform_list)):
        if transform_list[i] is not None:
            result = result * transform_list[i]
    return result
