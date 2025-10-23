import sys, math
import logging
import numpy as np
from ase.io import read, write
from os import environ, listdir, mkdir, chdir
from ase.calculators.dftb import Dftb

from ase.optimize import QuasiNewton, BFGS
import ase.calculators.mixing
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.units import kcal, fs
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from mace.calculators import MACECalculator

SPcalc = MACECalculator(model_path=sys.argv[1], device='cpu', default_dtype="float32")

xyz_pre = sys.argv[2]

#computing predicted property
logging.info("get predicted property")
flist = sorted(listdir(xyz_pre))

for iFile, fname in enumerate(flist):
    sysname = fname[:-4]
    atoms = read(xyz_pre+fname)

    DFTBcalc = Dftb(label='current_dftb',
                atoms=atoms,
                run_manyDftb_steps=True,
                Hamiltonian_SCC = 'Yes',
                Hamiltonian_MaxSCCIterations = '2000',
#                Hamiltonian_Filling = ' Fermi{ Temperature[K]= 50 }',
                Hamiltonian_ThirdOrderFull = 'Yes',
                Hamiltonian_SCCTolerance = '1E-6',
                Hamiltonian_PolynomialRepulsive_ = '',
                Hamiltonian_PolynomialRepulsive_setForAll = '{Yes}',
                Hamiltonian_Dispersion_ = 'MBD',
                Hamiltonian_Dispersion_KGrid = '1 1 1',
                Hamiltonian_Dispersion_Beta = 0.83, 
                Hamiltonian_Dispersion_NOmegaGrid = 25,
                Hamiltonian_Dispersion_ReferenceSet = 'ts',
                Analysis_ ='',
                Analysis_CalculateForces = 'Yes')

# Mixing calculators
    QMMMcalc =  ase.calculators.mixing.SumCalculator([DFTBcalc,SPcalc], atoms)

    atoms.set_calculator(QMMMcalc)

    qn = BFGS(atoms, trajectory='opt-mol.traj')
    qn.run(fmax=0.001,steps=4000)

    write("opt-mol.xyz", atoms, format='xyz')

    MaxwellBoltzmannDistribution(atoms, temperature_K=300, force_temp = True)

    dyn = Langevin(
        atoms = atoms,
        temperature_K = 300,
        friction = 2e-3
        timestep = 0.5*fs,
        fixcm = True, 
        logfile='md.log',
        trajectory='traj-md.traj',
        loginterval = 4
    )

    dyn.run(40000)

    traj = Trajectory("traj-md.traj")
    for atoms in traj:
        write("traj-md.xyz", atoms, append=True)
