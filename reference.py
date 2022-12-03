from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D
from conf_calc import ConfCalc

import numpy as np
import pandas as pd

from wilson_b_matrix import (
    Dihedral, 
    get_current_derivative,
    ext_current_derivative
)

def update_mol(molfile : str,
               parsed_coords : list[list[float]]):
    """
        Retruns mol with current coords
    """
    mol = Chem.MolFromMolFile(molfile, removeHs=False)
    for idx, coords in enumerate(parsed_coords):
        mol.GetConformer().SetAtomPosition(idx, Point3D(*coords))

    return mol

def read_grad(filename : str,
              number_of_atoms : int) -> list[list[float]]:
    """
        Read gradients from xtb output
    """

    grads = []
    with open(filename, 'r') as file:
        grads = [line[:-1] for line in file][(2 + number_of_atoms):-1]
    return list(map(lambda s: list(map(float, s.split())), grads))

def read_xyz(filename : str) -> list[list[float]]:
    """
        Reads .xyz file and parses to list of coords
    """
    coords = []
    with open(filename, 'r') as file:
        coords = [line[:-1] for line in file][2:]
    return list(map(lambda s: list(map(float, s.split()[1:])), coords))


def dihedral_angle(a : list[float], b : list[float], c : list[float], d : list[float]) -> float:
    """
        Calculates dihedral angel between 4 points
    """

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)

    #Next will be calc of signed dihedral angel in terms of rdkit
    #Vars named like in rdkit source code

    lengthSq = lambda u : np.sum(u ** 2)

    nIJK = np.cross(b - a, c - b)
    nJKL = np.cross(c - b, d - c)
    m = np.cross(nIJK, c - b)

    res =  -np.arctan2(np.dot(m, nJKL) / np.sqrt(lengthSq(m) * lengthSq(nJKL)),\
                       np.dot(nIJK, nJKL) / np.sqrt(lengthSq(nIJK) * lengthSq(nJKL)))
    return (res + 2 * np.pi) % (2 * np.pi)    

def get_dihedral_from_xyz(parsed_xyz : list[list[float]],
                          idxs : list[float]) -> float:
    """
        Calculates dihedral angle with current idxs
    """

    return dihedral_angle(*[parsed_xyz[idx] for idx in idxs])

class DerivativeResponse:

    def __init__(self, 
                 derivative : float,
                 point : list[float],
                 coords : list[list[float]],
                 grads : list[list[float]],
                 mol : Chem.rdchem.Mol):
        self.derivative = derivative
        self.point = point
        self.coords = coords
        self.grads = grads
        self.mol = mol

    def __repr__(self):
        return f"Derivative in point {self.point} is {self.derivative}.\n(If you want optimized structure coords or atoms' gradients in this point, call .coords or .grads fields)"

def get_der(calculator : ConfCalc, 
            vals : np.ndarray,
            theta_idx : int,
            rotable_idxs : list[list[int]],
            h : float = 1 * np.pi / 180) -> float:
    """
        Calculates derivative in point 'vals' with respect to 
        'theta_idx' angle. Derivative will be found by:
            f'(x) = (f(x + h) - f(x - h)) / 2h + O(h^2)
    """

    # optimise base point
    base_energy = calculator.get_energy(vals)
    ##print(f"Energy in base point: {base_energy}")
    #recalc first point
    opt_xyz = read_xyz("xtbopt.xyz")
    atom_grad = read_grad("gradient", len(opt_xyz))
    base_point = [get_dihedral_from_xyz(opt_xyz, cur_idxs) for cur_idxs in rotable_idxs]
    ##print(f"Base point become: {base_point}")
    opt_mol = update_mol("test.mol", opt_xyz)    

    #Calculator from optimized point
    opt_calc = ConfCalc(mol=opt_mol,
                        dir_to_xyzs="xtb_calcs/",
                        rotable_dihedral_idxs=rotable_idxs)

    #calc point (x + h)
    right_point = base_point.copy()
    right_point[theta_idx] += h
    right_energy = opt_calc.get_energy(right_point, req_opt=False)
    ##print(f"Energy in right point {right_point} is {right_energy}")

    #calc point(x - h)
    left_point = base_point.copy()
    left_point[theta_idx] -= h
    left_energy = opt_calc.get_energy(left_point, req_opt=False)
    ##print(f"Energy in left point {left_point} is {left_energy}")

    ##print(f"Derivative in {base_point} is {(right_energy - left_energy) / (2 * h)}")
    return DerivativeResponse((right_energy - left_energy) / (2 * h),
                              base_point, 
                              opt_xyz,
                              atom_grad, 
                              opt_mol)

def m(theta : float, v : np.ndarray) -> np.ndarray:
    return np.array([
        [np.cos(theta) + (1 - np.cos(theta)) * v[0] ** 2, 
         (1 - np.cos(theta)) * v[0] * v[1] - v[2] * np.sin(theta),
         (1 - np.cos(theta)) * v[0] * v[2] + v[1] * np.sin(theta)],
        [(1 - np.cos(theta)) * v[0] * v[1] + v[2] * np.sin(theta),
         np.cos(theta) + (1 - np.cos(theta)) * v[1] ** 2, 
         (1 - np.cos(theta)) * v[1] * v[2] - v[0] * np.sin(theta)],
        [(1 - np.cos(theta)) * v[0] * v[2] - v[1] * np.sin(theta),
         (1 - np.cos(theta)) * v[1] * v[2] + v[0] * np.sin(theta),
         np.cos(theta) + (1 - np.cos(theta)) * v[2] ** 2] 
    ])

def m_diff(theta : float, v : np.ndarray) -> np.ndarray:
    return np.array([
        [(v[0] ** 2 - 1) * np.sin(theta), 
         v[0] * v[1] * np.sin(theta) - v[2] * np.cos(theta),
         v[0] * v[2] * np.sin(theta) + v[1] * np.cos(theta)],
        [v[0] * v[1] * np.sin(theta) + v[2] * np.cos(theta),
         (v[1] ** 2 - 1) * np.sin(theta), 
         v[1] * v[2] * np.sin(theta) - v[0] * np.cos(theta)],
        [v[0] * v[2] * np.sin(theta) - v[1] * np.cos(theta),
         v[1] * v[2] * np.sin(theta) + v[0] * np.cos(theta),
         (v[2] ** 2 - 1) * np.sin(theta)] 
    ])

mol_file = "test.mol"
mol = Chem.MolFromMolFile(mol_file, removeHs=False)

rotable_idxs = [[5, 4, 6, 7],
                [4, 6, 7, 9],
                [12, 13, 15, 16]]

calculator = ConfCalc(mol=mol,
                      dir_to_xyzs="xtb_calcs/",
                      rotable_dihedral_idxs=rotable_idxs)

#print(get_der(calculator, np.zeros(3), 0, rotable_idxs))

data = pd.DataFrame(columns=["x1", "x2", "x3", "angle", "ab", "dc", "norm_cb", "dadt", "dddt", "theor_der", "ext_theor_der", "num_der"])

r = np.linspace(0, np.pi, 3)

for x_1 in r:
    for x_2 in r:
        for x_3 in r:
            for current_angle in range(3):
                response = get_der(calculator, np.array([x_1, x_2, x_3]),
                                   current_angle, rotable_idxs, 5 * np.pi / 180)
                ab = np.array(response.coords[rotable_idxs[current_angle][0]]) - np.array(response.coords[rotable_idxs[current_angle][1]])
                dc = np.array(response.coords[rotable_idxs[current_angle][3]]) - np.array(response.coords[rotable_idxs[current_angle][2]])
                norm_cb = np.array(response.coords[rotable_idxs[current_angle][2]]) - np.array(response.coords[rotable_idxs[current_angle][1]])
                norm_cb /= np.sqrt(norm_cb.T@norm_cb)
                dadt = m_diff(response.point[current_angle], norm_cb) @ m(response.point[current_angle], norm_cb).T @ ab
                dddt = m_diff(response.point[current_angle], norm_cb) @ m(response.point[current_angle], norm_cb).T @ dc
                ext_theor_der = ext_current_derivative(response.mol, np.array(response.grads).flatten(), Dihedral(*rotable_idxs[current_angle]))
                theor_der = get_current_derivative(response.mol, np.array(response.grads).flatten(), Dihedral(*rotable_idxs[current_angle]))
#                theor_der = np.sum(dadt * np.array(response.grads[rotable_idxs[current_angle][0]])) + np.sum(dddt * np.array(response.grads[rotable_idxs[current_angle][3]]))
                data = data.append({
                                    "x1" : response.point[0],
                                    "x2" : response.point[1],
                                    "x3" : response.point[2],
                                    "angle" : current_angle,
                                    "ab" : ab,
                                    "dc" : dc,
                                    "norm_cb" : norm_cb,
                                    "dadt" : dadt,
                                    "dddt" : dddt,
                                    "theor_der" : theor_der,
                                    "ext_theor_der" : ext_theor_der,
                                    "num_der" : response.derivative              
                                   }, ignore_index=True)
print(data)
data.to_csv("reference.csv")
