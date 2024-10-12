
To run an NEB simulation using our approach, follow these steps to execute the Python script.

```shell script
export OMP_NUM_THREADS=1
export DFTB_COMMAND='srun -N 1 dftb+'
export DFTB_PREFIX='./SKfiles/3ob-3-1/'

model=path_to_mace_model/
xyz=path_to_guess_trajectory/

python3 neb_script.py $model/MACE_model_swa.model $xyz
```
