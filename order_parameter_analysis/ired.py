import pymol
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import sympy
from pymol import cmd
import pickle


import numpy as np
import numpy.linalg as la
from MDAnalysis import Universe


def gen_cov(filename, atom1, atom2, length, shift=False):
    """
    params:
        filename: Path to the molecular structure file (e.g., PDB, GRO).
        atom1, atom2: Atom names (e.g., 'N', 'C', 'CA', 'CB', etc.).
        length: Expected number of residues in the structure.
        shift: Boolean, True when calculating shifted vectors (e.g., N-C across residues).
    """
    # Load the structure into MDAnalysis
    u = Universe(filename)

    # Extract coordinates for the specified atoms
    atom1_sel = u.select_atoms(f"resid 1:{length} and name {atom1}")
    atom2_sel = u.select_atoms(f"resid 1:{length} and name {atom2}")

    # Ensure both selections have matching residue indices
    atom1_resids = [atom.resid for atom in atom1_sel]
    atom2_resids = [atom.resid for atom in atom2_sel]
    index_atom1 = [i for i in range(1, length + 1) if i in atom1_resids]
    index_atom2 = [i for i in range(1, length + 1) if i in atom2_resids]

    index = sorted(set(index_atom1).union(set(index_atom2)))

    # Extract coordinates as numpy arrays
    atom1_coords = atom1_sel.positions
    atom2_coords = atom2_sel.positions

    if shift and atom1 == 'N' and atom2 == 'C':
        bond_vec = atom1_coords[1:] - atom2_coords[:-1]
        index = index_atom2[:-1]
    elif shift and atom1 == 'C' and atom2 == 'N':
        bond_vec = atom2_coords[1:] - atom1_coords[:-1]
        index = index_atom1[:-1]
    else:
        bond_vec = atom1_coords[:len(atom2_coords)] - atom2_coords
        index = index_atom2

    # Normalize bond vectors
    normalized_bond_vec = bond_vec / la.norm(bond_vec, axis=1).reshape(-1, 1)

    # Compute cosine similarity matrix
    cos = normalized_bond_vec @ normalized_bond_vec.T

    # Compute covariance matrix
    cov = 0.5 * (3 * cos**2 - 1)

    return np.array(index), cov

# Example Usage
# filename = "protein.pdb"
# atom1 = "N"
# atom2 = "C"
# length = 100
# indices, covariance_matrix = gen_cov(filename, atom1, atom2, length, shift=True)



# def gen_cov(filename,objname,atom1,atom2,length,pro=None,shift=False):
#     """
#     params: atom1,atom2: 'n','c','ca','cb','h'(nh)
#             shift: True when 'nc'
#     """
#     cmd.delete('all')
#     cmd.load(filename, objname)
    
#     atom1_all = cmd.get_coords('resi * and pol. and name {}'.format(atom1), 1)
#     atom2_all = cmd.get_coords('resi * and pol. and name {}'.format(atom2), 1)
#     index_atom1 = [i for i in range(length) if 
#                    cmd.get_coords('resi {} and pol. and name {}'.format(i+1,atom1), 1) is not None]
#     index_atom2 = [i for i in range(length) if 
#                    cmd.get_coords('resi {} and pol. and name {}'.format(i+1,atom2), 1) is not None]
    
#     index = sorted(list(set(index_atom1).union(set(index_atom2))))

#     if (shift and atom1=='n' and atom2=='c'):
#         bond_vec = (atom1_all[index][1:]-atom2_all[index][:-1])
#         index = index_atom2[:-1]
#     elif (shift and atom1=='c' and atom2=='n'):
#         bond_vec = (atom2_all[index][1:]-atom1_all[index][:-1])
#         index = index_atom1[:-1]
#     else:
#         index = index_atom2
#         bond_vec = (atom1_all[np.arange(length)[index_atom2]]-atom2_all)
        
#     normalized_bond_vec = bond_vec/la.norm(bond_vec,axis=1).reshape((len(bond_vec),-1))
    
#     cos = normalized_bond_vec @ normalized_bond_vec.T
#     cov = 0.5*(3*cos**2-1)
    
    
    
#     return np.array(index), cov


    
def block_ired(cov_ensemble,block_size,index,total_time=1000,M=5):
    i = 0
    s2 = []
    while i+block_size <= total_time:
        s2_tmp = []
        angle = np.array(cov_ensemble[i:i+block_size]).mean(axis=0)
        w, v = la.eig(angle)
        for k in range(len(index)):
            sk = []
            for m in range(M,len(index)):
                sk.append(w[m]*v[k,m]**2)
            s2_tmp.append(1-sum(sk))
        s2.append(s2_tmp) 
        i += block_size
    s2_averaged = np.array(s2).mean(axis=0)
    std = np.array(s2).std(axis=0)
    return np.array(s2_averaged),np.array(s2),np.array(std),