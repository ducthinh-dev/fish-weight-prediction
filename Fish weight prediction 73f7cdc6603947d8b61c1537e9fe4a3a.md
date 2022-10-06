# Fish weight prediction

> By: Do Duc Thinh — [dataset](https://www.kaggle.com/datasets/aungpyaeap/fish-market) — [notebook](https://colab.research.google.com/drive/1W9XbDFf-s40jTee-TM_nr9LmSeXbNLeX?usp=sharing)
> 

<aside>
💡 This report is made for the final project of ****Supervised Machine Learning: Regression**** on [Coursera](https://www.coursera.org/learn/supervised-machine-learning-regression)

</aside>

# Dataset & objectives

## Dataset

### Data source

The dataset is published on Kaggle and can be accessed [here](https://www.kaggle.com/datasets/aungpyaeap/fish-market).

### Dataset description

This dataset is a record of 7 common different fish species in fish market sales. With this dataset, a predictive model can be performed using machine-friendly data, and estimate the weight of fish can be predicted.

## Objective

The main objective is to predict the weight of fish. Then optimizing the model for better predictions.

The model training will feature three linear regression models using simple linear regression as a baseline.

# Data exploration & cleaning

## Exploratory data analysis

### Data summary

The dataset contains 7 columns:

- Species: Species name of fish
- Weight: Weight of fish in gram
- Length1: Vertical length in cm
- Length2: Diagonal length in cm
- Length3: Cross length in cm
- Height: Height in cm
- Width: Diagonal width in cm

The shape of the dataset is `(159, 7)`.

First 5 rows of the dataset:

| --- | --- | --- | --- | --- | --- | --- | --- |

Types of features:

```python
Species     object
Weight     float64
Length1    float64
Length2    float64
Length3    float64
Height     float64
Width      float64
dtype: object
```

Summary statistics:

| --- | --- | --- | --- | --- |

```python
> df.isnull().sum()
Species    0
Weight     0
Length1    0
Length2    0
Length3    0
Height     0
Width      0
dtype: int64
```

There are no missing values found.

### Analysis of “Species”

![download.png](Fish%20weight%20prediction%2073f7cdc6603947d8b61c1537e9fe4a3a/download.png)

There are 7 different species in which *Perch* has the maximum count and Whitefish has the minimum count.

### Analysis of target feature - Weight

![download.png](Fish%20weight%20prediction%2073f7cdc6603947d8b61c1537e9fe4a3a/download%201.png)

The target feature *Weight* has linear relationships with all the other variables.

![download.png](Fish%20weight%20prediction%2073f7cdc6603947d8b61c1537e9fe4a3a/download%202.png)

There is a problem with multi-colinearity, to deal with this only one length variable will be used. Which length variable is used is decided later.

![download.png](Fish%20weight%20prediction%2073f7cdc6603947d8b61c1537e9fe4a3a/download%203.png)

The above plot shows that fish weight has a heavy right skew distribution. While the other variables are more normally distributed with slight right tails.

## Cleaning & feature selection

### Outlier analysis

Using visualization to detect outlier values.

![download.png](Fish%20weight%20prediction%2073f7cdc6603947d8b61c1537e9fe4a3a/download%204.png)

Making a function to check outlier values for each feature

```python
def detect_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)
    outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)]
    
    if len(outliers) == 0:
        return 'No Outliers Present'
    else:
        return outliers

for col in numerical_features:
    print('-' * 27, col,'-' * 26)
    print(detect_outliers(df, col))
    print('\n')
```

The return results:

```python
--------------------------- Weight --------------------------
    Species  Weight  Length1  Length2  Length3  Height  Width
142    Pike  1600.0     56.0     60.0     64.0   9.600  6.144
143    Pike  1550.0     56.0     60.0     64.0   9.600  6.144
144    Pike  1650.0     59.0     63.4     68.0  10.812  7.480

--------------------------- Length1 --------------------------
    Species  Weight  Length1  Length2  Length3  Height  Width
142    Pike  1600.0     56.0     60.0     64.0   9.600  6.144
143    Pike  1550.0     56.0     60.0     64.0   9.600  6.144
144    Pike  1650.0     59.0     63.4     68.0  10.812  7.480

--------------------------- Length2 --------------------------
    Species  Weight  Length1  Length2  Length3  Height  Width
142    Pike  1600.0     56.0     60.0     64.0   9.600  6.144
143    Pike  1550.0     56.0     60.0     64.0   9.600  6.144
144    Pike  1650.0     59.0     63.4     68.0  10.812  7.480

--------------------------- Length3 --------------------------
    Species  Weight  Length1  Length2  Length3  Height  Width
144    Pike  1650.0     59.0     63.4     68.0  10.812   7.48

--------------------------- Height --------------------------
No Outliers Present

--------------------------- Width --------------------------
No Outliers Present
```

All the outliers of the dataset line in rows 142 to 144. Then drop outliers.

### Feature selection

Multi-collinearity detection

```python
X_collinearity = df[df.columns[2:]]
y_collinearity = df['Weight']
X_sm = sm.add_constant(X_collinearity)
sm_model = sm.OLS(y_collinearity,X_sm).fit()

print(sm_model.summary())
```

The return result:

```python
														OLS Regression Results                            
==============================================================================
Dep. Variable:                 Weight   R-squared:                       0.904
Model:                            OLS   Adj. R-squared:                  0.901
Method:                 Least Squares   F-statistic:                     282.0
Date:                Mon, 03 Oct 2022   Prob (F-statistic):           2.27e-74
Time:                        06:45:38   Log-Likelihood:                -937.83
No. Observations:                 156   AIC:                             1888.
Df Residuals:                     150   BIC:                             1906.
Df Model:                           5                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       -426.9434     25.576    -16.693      0.000    -477.479    -376.408
Length1      102.5579     33.218      3.087      0.002      36.923     168.193
Length2      -45.0763     34.413     -1.310      0.192    -113.073      22.920
Length3      -37.1475     14.293     -2.599      0.010     -65.388      -8.907
Height        36.8466      7.254      5.079      0.000      22.513      51.180
Width         52.4022     17.113      3.062      0.003      18.588      86.217
==============================================================================
Omnibus:                       12.858   Durbin-Watson:                   0.527
Prob(Omnibus):                  0.002   Jarque-Bera (JB):               14.260
Skew:                           0.741   Prob(JB):                     0.000801
Kurtosis:                       2.994   Cond. No.                         307.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

There are some high p-values. Some variables are insignificant. Let's calculate VIF. A variance inflation factor(VIF) detects multicollinearity in regression analysis.

```python
def VIF():
    vif = pd.DataFrame()
    vif['features'] = X_collinearity.columns
    vif['VIF'] = [ variance_inflation_factor(X_collinearity.values, i) for i in range(X_collinearity.shape[1])]
    
    vif['VIF'] = round(vif['VIF'] ,2)
    vif = vif.sort_values(by='VIF',ascending = False)
    return vif

vif = VIF()
vif
```

Return result:

| --- | --- | --- |

From the Correlation plot, OLS Method, and VIF, it concludes that to overcome multi-collinearity, we'll have to remove highly correlated independent variables (Length2 and Length3).

# Data preprocessing & model selection

## Data preprocessing

### Features preprocessing

First split the data into X and y. Categorical and numeric pipelines will be created for categorical and numerical features to transform.

- The categorical features will be encoded with One-Hot encoding.
- For the numeric features, StandardScaler will be used so that the features have 0 mean and a variance of 1.

```python
X = df.drop('Weight', axis=1)
y = df['Weight']

categorical_pipeline = Pipeline(
    steps=[
        ("onehot-encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

numeric_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

# categorical and numeric features
cat_cols = X.select_dtypes(exclude="number").columns
num_cols = X.select_dtypes(include="number").columns
```

The features will be inputted along with their corresponding pipelines into a ColumnTransformer instance.

```python
full_processor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, num_cols),
        ("categorical", categorical_pipeline, cat_cols),
    ]
)

X_processed = full_processor.fit_transform(X)
y_processed = StandardScaler().fit_transform(y.values.reshape(-1, 1))
```

### Train & test split

```python
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=2022)
```

## Model selection

### Model evaluation

Model evaluation function includes Mean squared error, Mean absolute error, and R2 score.

```python
def evaluate_model(y_pred, y_test):
    MSE = round(mean_squared_error(y_test, y_pred),2)
    MAE = round(mean_absolute_error(y_test, y_pred),2) 
    r2score = round(r2_score(y_test, y_pred),2) 
    return MSE, MAE, r2score
```

### Models

Models with default parameters.

```python
models = [ LinearRegression(),
           Ridge(alpha=0.1),
           Lasso(alpha=0.1)]

model_names = ['Linear Regression','Ridge','Lasso']

def build_models(models, model_names):
    mse  = []
    mae = []
    r2 = []
    results = {}
    
    for idx, (ml_model_names, ml_models) in enumerate(zip(model_names, models)):
        reg = models[idx]
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        MSE, MAE, r2score = evaluate_model(y_test, y_pred)
        mse.append(MSE)
        mae.append(MAE)
        r2.append(r2score)
    
    results = {'Model':model_names,
           'MSE':mse,
           'MAE':mae,
           'R2 Score':r2}
    models_scores_df = pd.DataFrame(results)
    return models_scores_df
```

Apply the above function to see the results

```python
models_scores_df = build_models(models, model_names)
models_scores_df
```

| --- | --- | --- | --- | --- |

The final results show that both baseline linear regression and ridge regression models have about the same scores while the lasso model underperforms. They can be used as the final model but I personally recommend using the ridge model.

# Conclusion

## Summary & insights

The fish's weight heavily depends on its length and is less likely on other attributes. Fish

## Further suggestions

For better prediction, I personally suggest adding features like habitat, location, and age.