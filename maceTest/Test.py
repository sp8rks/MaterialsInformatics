# Description: Test the MACE code on a simple system
from mace.calculators import mace_mp
from ase.md.langevin import Langevin
from ase.io import read, write
from ase.units import fs, kB

fname = '/Users/stanleywessman/Downloads/MaterialsInformatics/testStuff/n-pentane.xyz'
atoms = read(fname)
calc = mace_mp()

atoms.set_calculator(calc)

dyn = Langevin(atoms, fs, kB * 700, 0.002)
dyn.attach(write, 1, 'md.xyz', atoms)