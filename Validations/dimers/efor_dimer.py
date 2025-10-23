import torch
import logging
import numpy as np
#import ase
import sys
from ase.io import read, write
from ase.db import connect
from ase.data.s66x8 import s66x8
from ase.calculators.dftb import Dftb
from ase.units import Hartree, Bohr, kcal, mol
from ase.io.extxyz import write_extxyz
from ase import Atoms

from mace.calculators import MACECalculator

MLcalc = MACECalculator(model_path=sys.argv[1], device='cpu', default_dtype="float32")

o1 = open('output-s66x8-tot.dat', 'w')
o2 = open('output-s66x8-mono1.dat', 'w')
o3 = open('output-s66x8-mono2.dat', 'w')
o4 = open('output-s66x8-Eint.dat', 'w')

for name in s66x8.get_names():
    atoms = s66x8.create_s66x8_system(name)

    ## set up calculator as you wish...
    DFTBcalc = Dftb(label='current_dftb',
                atoms=atoms,
                Hamiltonian_SCC = 'Yes',
                Hamiltonian_ThirdOrderFull = 'Yes',
                Hamiltonian_HCorrection_ = 'Damping',
                Hamiltonian_HCorrection_Exponent =4.05,
                Hamiltonian_PolynomialRepulsive_ = '',
                Hamiltonian_PolynomialRepulsive_setForAll = '{Yes}',
                Hamiltonian_Dispersion_ = 'MBD',
                Hamiltonian_Dispersion_KGrid = '1 1 1',
                Hamiltonian_Dispersion_Beta = 0.83,
                Hamiltonian_Dispersion_NOmegaGrid = 25,
                Hamiltonian_Dispersion_ReferenceSet = 'ts',
                ParserOptions_ParserVersion = '13')

    atoms.set_calculator(DFTBcalc)

    elenergy = float(atoms.get_total_energy())
    elftb = np.array(atoms.get_forces())

    atoms.set_calculator(MLcalc)

    PAE = float(atoms.get_total_energy())
    pftb = np.array(atoms.get_forces())

    edim  = elenergy + PAE
    totdftb = elftb + pftb

    R = atoms.get_positions()
    Z = atoms.get_atomic_numbers()
    dim = Atoms(Z, R)

    dim.info['energy'] = edim
    dim.arrays['forces'] = np.array(totdftb)

    write_extxyz(name+'.xyz', dim)

    o1.write("{:>30}".format(str(name)) + "{:>24}".format(elenergy) + "{:>24}".format(PAE) + "{:>24}".format(edim) + "\n")

    mono1 = s66x8.create_s66x8_monomer1(name)

    ## set up calculator as you wish...
    DFTBcalc = Dftb(label='current_dftb',
                atoms=mono1,
                Hamiltonian_SCC = 'Yes',
                Hamiltonian_ThirdOrderFull = 'Yes',
                Hamiltonian_HCorrection_ = 'Damping',
                Hamiltonian_HCorrection_Exponent =4.05,
                Hamiltonian_PolynomialRepulsive_ = '',
                Hamiltonian_PolynomialRepulsive_setForAll = '{Yes}',
                Hamiltonian_Dispersion_ = 'MBD',
                Hamiltonian_Dispersion_KGrid = '1 1 1',
                Hamiltonian_Dispersion_Beta = 0.83, 
                Hamiltonian_Dispersion_NOmegaGrid = 25,
                Hamiltonian_Dispersion_ReferenceSet = 'ts',
                ParserOptions_ParserVersion = '13')

    mono1.set_calculator(DFTBcalc)

    elenergy = float(mono1.get_total_energy())
    elftb1 = np.array(mono1.get_forces())

    mono1.set_calculator(MLcalc)

    PAE1 = float(mono1.get_total_energy())
    pftb1 = np.array(mono1.get_forces())

    emon1 = elenergy  + PAE1
    totdftb1 = elftb1 + pftb1

    R1 = mono1.get_positions()
    Z1 = mono1.get_atomic_numbers()
    m1 = Atoms(Z1, R1)

    m1.info['energy'] = emon1
    m1.arrays['forces'] = np.array(totdftb1)

    write_extxyz(name+'-mono1.xyz', m1)

    o2.write("{:>30}".format(str(name)) + "{:>24}".format(elenergy) + "{:>24}".format(PAE1) + "{:>24}".format(emon1) + "\n")

    mono2 = s66x8.create_s66x8_monomer2(name)

    ## set up calculator as you wish...
    DFTBcalc = Dftb(label='current_dftb',
                atoms=mono2,
                Hamiltonian_SCC = 'Yes',
                Hamiltonian_ThirdOrderFull = 'Yes',
                Hamiltonian_HCorrection_ = 'Damping',
                Hamiltonian_HCorrection_Exponent =4.05,
                Hamiltonian_PolynomialRepulsive_ = '',
                Hamiltonian_PolynomialRepulsive_setForAll = '{Yes}',
                Hamiltonian_Dispersion_ = 'MBD',
                Hamiltonian_Dispersion_KGrid = '1 1 1',
                Hamiltonian_Dispersion_Beta = 0.83,
                Hamiltonian_Dispersion_NOmegaGrid = 25,
                Hamiltonian_Dispersion_ReferenceSet = 'ts',
                ParserOptions_ParserVersion = '13')

    mono2.set_calculator(DFTBcalc)

    elenergy = float(mono2.get_total_energy())
    elftb2 = np.array(mono2.get_forces())

    mono2.set_calculator(MLcalc)

    PAE2 = float(mono2.get_total_energy())
    pftb2 = np.array(mono2.get_forces())

    emon2 = elenergy + PAE2
    totdftb2 = elftb2 + pftb2

    R2 = mono2.get_positions()
    Z2 = mono2.get_atomic_numbers()
    m2 = Atoms(Z2, R2)

    m2.info['energy'] = emon2
    m2.arrays['forces'] = np.array(totdftb2)

    write_extxyz(name+'-mono2.xyz', m2)

    o3.write("{:>30}".format(str(name)) + "{:>24}".format(elenergy) + "{:>24}".format(PAE2) + "{:>24}".format(emon2) + "\n")

    Eint = edim - emon1 - emon2
#    int1 = s66x8.get_interaction_energy_CC(name)
#    int2 = s66x8.get_interaction_energy_PBE_MBD(name)*kcal/mol
#    int3 = s66x8.get_interaction_energy_PBE_TS(name)*kcal/mol

    o4.write("{:>30}".format(str(name))  + "{:>24}".format(Eint) + "\n")
  
o1.close()
o2.close()
o3.close()
o4.close()
