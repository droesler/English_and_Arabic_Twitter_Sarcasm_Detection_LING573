## How Set up Environment and Replicate Results for D2 on patas

We use conda to manage our dev environment. To rerun our evaluation script, you can run the following command to set up the environment first:

```
wget 
bash Anaconda3-2021.11-Linux-x86_64.sh
```


To replicate the result:


### Baseline
We use random forest and lightgbm to train our baseline models. You can view the results in `/results/D2/baseline_rf` and `/results/D2/baseline_lightgbm` or replicate the results by running the following command. Make sure you are under directory `/src/baseline`

```
# Baseline Models
cd src/baseline
python test_models.py
```

# Evaluate best model for D2

```
cd ../..   # go back to root folder
python src/model_runner.py predict input_file pred_output_file --log-dir .logging --experiment-name prod --experiment-version 1
```
