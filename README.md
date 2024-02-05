# Instructions on Running the Code

### Preparation Step
In this step, we generate the model class.
```
python Preparation.py --M-size 200 --scale 0.1 -S 100 -A 50 -H 3 --d-phi 5 --d-psi 5
```

### Running Step
Command line for `run.py`.
```
python run.py --M-size 200 --scale 0.1 -S 100 -A 50 -H 3 --d-phi 5 --d-psi 5 --scale 0.1 -T 50 --delta 0.001 --model-seed 0 1000 ... --seed 0 1000 ...
```


### Visualize Results
By setting the argument of `--key`, we can visualize the learning curves regarding the size of remaining models (`M_size`) or the ratio between the NE-Gap of the estimated policy and the maximal gap (`ratio`).
```
python plot.py --key M_size
python plot.py --key ratio
```

Before plotting the curves, please change the `dirs` variable in the `main()` function in `plot.py`.