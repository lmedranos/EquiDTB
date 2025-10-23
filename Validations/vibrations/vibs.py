import sys
import os
from os import path
from ase.io import read, write
from os import environ, listdir, mkdir, chdir, system, remove
from ase import Atoms
import numpy as np
from ase.optimize import QuasiNewton, BFGS
from ase.vibrations import Vibrations
from ase.calculators.dftb import Dftb
from ase.units import Hartree, Bohr, kcal, mol
import ase.calculators.mixing

from mace.calculators import MACECalculator

SPcalc = MACECalculator(model_path=sys.argv[1], device='cpu', default_dtype="float32")

xyz_pre = sys.argv[2]
ofile = sys.argv[3]

folders = sorted(listdir(xyz_pre))
chdir(ofile)

for fdir in folders:
  fname = fdir[:-4]

  if path.exists(ofile+fname) == False:
    mkdir(ofile+fname)
  chdir(ofile+fname)

  atoms = read(xyz_pre+fdir, format='xyz')

  ## set up calculator as you wish...
  DFTBcalc = Dftb(label='current_dftb',
                atoms=atoms,
                Hamiltonian_SCC = 'Yes',
                Hamiltonian_ThirdOrderFull = 'Yes',
                Hamiltonian_SCCTolerance = '1E-6',
#                Hamiltonian_Filling = ' Fermi{ Temperature[K]= 50 }',
                Hamiltonian_PolynomialRepulsive_ = '',
                Hamiltonian_PolynomialRepulsive_setForAll = '{Yes}',
                Hamiltonian_Dispersion_ = 'MBD',
                Hamiltonian_Dispersion_KGrid = '1 1 1',
                Hamiltonian_Dispersion_Beta = 0.83, 
                Hamiltonian_Dispersion_NOmegaGrid = 25,
                Hamiltonian_Dispersion_ReferenceSet = 'ts',
                Analysis_ ='',
                Analysis_CalculateForces = 'Yes')

  QMMMcalc =  ase.calculators.mixing.SumCalculator([DFTBcalc,SPcalc], atoms)

  atoms.set_calculator(QMMMcalc)

  try:
    qn = BFGS(atoms, trajectory='mol.traj')
    qn.run(fmax=0.0001, steps=4000)
  except Exception:
    print('|| ERROR IN INITIAL ENERGY (IO)!  for '+str(fname))
    system('rm band.out  charges.bin  current_dftb.out  detailed.out  dftb_in.hsd  dftb_pin.hsd  geo_end.gen result.tag')
    continue

  write('opt-'+str(fname)+'.xyz', atoms)

  vib = Vibrations(atoms, delta=0.01,nfree=4)
  vib.run()
  vib.summary()

  vb_ev = vib.get_energies()
  vb_cm = vib.get_frequencies()
  ENE = atoms.get_total_energy()

  o1 = open('info-modes-'+str(fname)+'.dat', 'w')
  o1.write("{: >24}".format(fname)  +  "{: >24}".format(ENE) + "\n")

  for i in range(0, len(vb_ev)):
    o1.write("{: >24}".format(vb_ev.real[i]) + "{: >24}".format(vb_ev.imag[i]) + "{: >24}".format(vb_cm.real[i]) + "{: >24}".format(vb_cm.imag[i]) + "\n")

  vib.write_jmol()
  o1.close()
