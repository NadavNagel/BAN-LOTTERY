# BAN-LOTTERY
Born Again as a Sparse Lottery Wining Networks

## Training
```bash
$ python train.py --h

optional arguments:
  --weight WEIGHT
  --lr Lerning Rate
  --batch_size BATCH_SIZE
  --dataset DATASET 
  --outdir OUTDIR
  --print_interval PRINT_INTERVAL every X iteretoins
  --decay DECAY for learning rate
  --experiment_name CHOOSE EXPERIMENT NAME 
  --prune CHOOSE PRUNE TIMIMNG :begining_after/every_gen/begining_before/None.
      begining_after - prune only after first generation learning.
      begining_before - prune only at the beging of the first student learning (before start lerned)
      every_gen - prune after every student
      None - Do not prune
  --init_teq  random/fix/increase/decrese
  --n_epoch Number of epoch for every student
  --n_gen Number of student
  --label soft/hard
      soft - Use soft and hard labels
      hard - Use only hard label
  --window 0/1/2/3 How many last students results take into account

```

## Infer
```bash
$ python infer.py --h

optional arguments:
  --dataset DATASET 
  --weights_root OUTDIR
  --batch_size BATCH_SIZE 
  --experiment_name 
  --ensemble_models list of generetuin you want to calculate the ensemble with

```
