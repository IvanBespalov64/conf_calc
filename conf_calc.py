import numpy as np

from typing import Tuple, Union

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from wilson_b_matrix import (
    Dihedral,
    get_current_derivative
)

import copy
import os
import time

class ConfCalc:

    def __init__(self, 
                 mol_file_name : str = None,
                 mol : Chem.Mol = None,
                 rotable_dihedral_idxs : list[list[int]] = None,
                 dir_to_xyzs : str = "", 
                 charge : int = 0,
                 gfn_method : int = 2,
                 timeout : int = 250,
                 norm_en : int = 0.):
        """
            Class that calculates energy of current molecule
            with selected values of dihedral angles
            mol_file_name - path to .mol file
            mol - Chem.Mol object
            rotable_dihedral_idxs - list of 4-element-lists with 
            integer zero-numerated indexes
            dir_to_xyzs - path of dir, where .xyz files will be saved
            charge - charge of molecule
            gfn_method - type of GFN method
            timeout - period of checking log file
            norm_en - norm energy 
        """        
        
        assert (mol_file_name is not None or mol is not None),\
             """No mol selected!"""

        assert rotable_dihedral_idxs is not None,\
             """No idxs to rotate have been selected!"""

        if mol_file_name is None:
            self.mol = mol
        elif mol is None:
            self.mol = Chem.MolFromMolFile(mol_file_name, removeHs=False)  

        self.rotable_dihedral_idxs = rotable_dihedral_idxs
        
        if dir_to_xyzs != "":
            dir_to_xyzs = dir_to_xyzs if dir_to_xyzs[-1] == "/" else dir_to_xyzs + "/"

        self.dir_to_xyzs = dir_to_xyzs

        self.charge = charge
        self.gfn_method = gfn_method
        self.timeout = timeout
        self.norm_en = norm_en

        # Id of next structure to save
        self.current_id = 0

    def set_norm_en(self,
                    norm_en = 0.):
        """
            Updates norm energy
        """

        self.norm_en = norm_en

    def __setup_dihedrals(self,
                          values : list[float]) -> Chem.Mol:
        """
            Private function that returns a molecule with
            selected dihedral angles
            values - list of angles in radians
        """
        
        assert len(values) == len(self.rotable_dihedral_idxs),\
             """Number of values must be equal to the number of
                dihedral angles"""
        
        new_mol = copy.deepcopy(self.mol)        

        for idxs, value in zip(self.rotable_dihedral_idxs, values):
            rdMolTransforms.SetDihedralRad(new_mol.GetConformer(), *idxs, value)
        
        return new_mol

    def __save_mol_to_xyz(self, 
                          mol : Chem.Mol) -> str:
        """
            Saves given mol to file, returns name of file
        """
        
        file_name = self.dir_to_xyzs + str(self.current_id) + ".xyz"
        self.current_id += 1
        Chem.MolToXYZFile(mol, file_name)

        return file_name

    def __generate_inp(self, 
                       values : list[float]) -> str:
        """
            Generate input that constrains values
            of dihedral angles from 'vals'
            Returns name of inp file
        """

        inp_name = self.dir_to_xyzs + str(self.current_id) + ".inp"
        
        with open(inp_name, "w+") as inp_file:
            inp_file.write("$constrain\n")
            for idxs, value in zip(self.rotable_dihedral_idxs, values):
                inp_file.write(f"dihedral: {idxs[0] + 1}, {idxs[1] + 1}, {idxs[2] + 1}, {idxs[3] + 1}, {180 * value / np.pi}\n")
            inp_file.write("$end") 
        return inp_name

    def __run_xtb(self,
                  xyz_name : str,
                  inp_name : str,
                  req_opt : bool = True,
                  req_grad : bool = True) -> Union[str, None]:
        """
            Runs xtb with current xyz_file, returns name of log file
            Args:
                xyz_name - name of .xyz file
                inp_name - name of .inp file
                req_opt - optimization or single-point energy
                req_grad - gradients required
            Returns:
                logfile name if optimization succeeded, None otherwise

        """

        log_name = xyz_name[:-3] + "log"
        os.system(f"xtb --input {inp_name} --charge {self.charge} --gfn {self.gfn_method} {xyz_name} {'--opt' if req_opt else ''} {'--grad' if req_grad else ''} > {log_name}")
        
        while True:
            try:
                with open(log_name, "r") as file:
                    file_lines = [line for line in file]    

                    line_with_en = [
                        line for line in file_lines
                        if "TOTAL ENERGY" in line
                    ]

                    line_with_error = [
                        line for line in file_lines
                        if "[ERROR] Program stopped due to fatal error" in line
                    ]

                    if len(line_with_en) != 0:
                        return log_name
                    if len(line_with_error) != 0:
                        return None
            except FileNotFoundError:
                pass
            finally:
                time.sleep(self.timeout / 1000)

    def __parse_energy_from_log(self, 
                                log_name : str) -> float:
        """
            Gets energy from xtb log file
        """

        energy = 0.
        with open(log_name, "r") as log_file:
            energy = [
                line for line in log_file 
                    if "TOTAL ENERGY" in line
                ][0].split()[3]

        return float(energy)

    def __parse_grads_from_grads_file(self,
                                      num_of_atoms : int,
                                      grads_filename : str = "gradient") -> np.ndarray:
        """
            Read gradinets from file, returns ['num_of_atoms', 3] numpy array with
            cartesian energy derivatives
        """
        grads = []
        with open(grads_filename, 'r') as file:
            grads = [line[:-1] for line in file][(2 + num_of_atoms):-1]
        return np.array(list(map(lambda s: list(map(float, s.split())), grads)))            

    def __calc_energy(self, 
                      mol : Chem.Mol, 
                      inp_name : str,
                      req_opt : bool = True,
                      req_grad : bool = True) -> Tuple[float, float]:
        """
            Calculates energy of given molecule via xtb
            inp_name - name of file with input
            retruns tuple of energy and gradient
        """
        xyz_name = self.__save_mol_to_xyz(mol)
        log_name = self.__run_xtb(xyz_name, 
                                  inp_name, 
                                  req_opt=req_opt,
                                  req_grad=req_grad)
        
        if log_name is None:
            return None, None
        
        irc_grad = None
        if req_grad:
            cart_grads = self.__parse_grads_from_grads_file(len(mol.GetAtoms()))
            irc_grad = []
            for rotable_idx in self.rotable_dihedral_idxs:
                irc_grad.append((
                        rotable_idx, 
                        get_current_derivative(mol,
                                               cart_grads.flatten(),
                                               Dihedral(*rotable_idx))    
                ))
        
        return self.__parse_energy_from_log(log_name), irc_grad

    def get_energy(self, 
                   values : list[float], 
                   req_opt : bool = True,
                   req_grad : bool = True) -> float:
        """
            Returns dict with fields:
            'energy' - energy in this point
            'grads' - list of tuples, consists of 
            pairs of dihedral angle atom indexes and 
            gradients of energy with resoect to this angle
        """
        inp_name = self.__generate_inp(values)
        mol = self.__setup_dihedrals(values)
        
        energy, grad = self.__calc_energy(mol, 
                                          inp_name, 
                                          req_opt=req_opt,
                                          req_grad=req_grad)
        if energy:
            energy -= self.norm_en

        return {    
            'energy' : energy,
            'grads' : grad
           }
    

