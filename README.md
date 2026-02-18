## SEMI-CONDUCTOR MANUFACTURING PROCESS PREDICTION/FAULT DETECTION
> It's a dataset from a real semiconductor manufacturing process where the goal is yield prediction / fault detection — figuring out which wafers pass or fail quality control based on hundreds of sensor readings.
> 
>  Data Source : https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data

### PROJECT OVERVIEW
1. 590 sensor signals collected from the wafer fabrication line
2. Most signals are noisy, irrelevant, or correlated with each other
3. Goal: find the small subset of signals that actually predict pass/fail (the -1 and 1 labels in the target file)
4. Real business impact: reduce scrap rate, cut costs, increase throughput.


#### Pipeline Practice Imputation for Producion Level ML
```angular2html
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("model", LogisticRegression())
])

pipeline.fit(X_train_1, y_train)

```


N/B: Important step in data pre-processing and EDA is checking for constant features in columns.
- It is also important to check at the behaviour of null values and even  plot it in a  msno matrix
- It is also a best practice to check for outliers after splitting the data, it should never influence the testing data.

