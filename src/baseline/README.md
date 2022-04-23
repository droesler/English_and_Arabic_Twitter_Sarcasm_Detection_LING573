## Baseline Models

After setting up conda environment and installing all the dependencies, run `python test_model.py` to produce the results. 

Random Forest Result:
```
                precision    recall  f1-score   support

           0       0.55      0.41      0.47        86
           1       0.53      0.66      0.59        86

    accuracy                           0.53       172
   macro avg       0.54      0.53      0.53       172
weighted avg       0.54      0.53      0.53       172
```

LightGBM Result:
```
                precision    recall  f1-score   support

           0       0.55      0.30      0.39        86
           1       0.52      0.76      0.62        86

    accuracy                           0.53       172
   macro avg       0.54      0.53      0.50       172
weighted avg       0.54      0.53      0.50       172
```