import sys, os
from ase.neb import NEB,NEBOptimizer, NEBTools
from ase.optimize import BFGS, LBFGS, QuasiNewton
import ase
from ase import Atoms
from ase.io import write, read
from ase.calculators.dftb import Dftb
from os import listdir
import ase.calculators.mixing
import numpy as np
import rmsd
import math

def get_ordered_molecules(path_to_folder):
    
    angles = []
    list_of_names = listdir(path_to_folder)
    for name in list_of_names:
        tmp = name[0:-7]
        angles.append(float(tmp[-3::]))
    angles = np.array(angles)
    sorted_a = np.argsort(angles)
    sorted_list = [list_of_names[i] for i in sorted_a]
    images = []
    for name in sorted_list:
        images.append(read(path_to_folder + name))
    return images, angles

from mace.calculators import MACECalculator

SPcalc = MACECalculator(model_path=sys.argv[1], device='cpu', default_dtype="float32")

path_to_folder = sys.argv[2]

#get images and angles (angles are useless but who knows)
images, angles = get_ordered_molecules(path_to_folder)

#get the position of the first image
pos_old = images[0].get_positions()

#recenter all the others based on this
for i in range(0,len(images)):    
    image = images[i]
    pos = image.get_positions()
    pos = pos - rmsd.centroid(pos)
    rotation = rmsd.quaternion_rotate(pos, pos_old)
    pos = pos@rotation
    images[i].set_positions(pos)

#for initial and final we reoptimize with the constraint before the neb optimization (remove calculator at the end)
for i in [0,-1]:

    DFTBcalc = Dftb(label='current_dftb',
                atoms=images[i],
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
                Analysis_CalculateForces = 'Yes')#,

    calc =  ase.calculators.mixing.SumCalculator([DFTBcalc,SPcalc], images[i])

    images[i].calc = calc

    opt = BFGS(images[i])
    opt.run(fmax = 0.001, steps = 2000)

    images[i].set_calculator()

#you will need to have a different calculator per image
# the first and last image are fixed, no calculator please
for i in range(1,len(images)-1):

    DFTBcalc = Dftb(label='current_dftb',
                atoms=images[i],
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
                Analysis_CalculateForces = 'Yes')#,

    calc =  ase.calculators.mixing.SumCalculator([DFTBcalc,SPcalc], images[i])

    images[i].calc = calc

#define the neb object with the list of molecules (list of Atoms objects) 
neb = NEB(images, remove_rotation_and_translation=True)

#define the optimizer for the NEB, you can choose another one like BFGS if you like
optimizer = NEBOptimizer(neb)

#run the optimization (the specifics steps and fmax are useless for NEBOptimizer but work for all the others)
optimizer.run(fmax = 0.005, steps = 4000)

#plot the energy curve, we need to add a calculator also to initial and final geometry
images[0].set_calculator(calc)
images[-1].set_calculator(calc)

o1 = open('energy-neb.dat', 'w')
for ii, image in enumerate(images):
    ener = float(image.get_total_energy())
    write('pos-'+str(ii)+'.xyz', image, format='xyz')
    o1.write("{:>24}".format(ener) + "\n")

