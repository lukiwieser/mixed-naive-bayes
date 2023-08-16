# Mixed Naive Bayes

Naive Bayes Classifier for categorical and/or numeric features.

Scikit learn currently only supports numeric data (with GaussianNB) or categorical data (with CategoricalNB), but not both. 
That's why I created this prototype.

## Usage

This project uses Python 3.9+.

First install the dependencies with:

```console
pip install -r requirements.txt
```

Then simply import the class `MixedNB` from `naive_bayes.py`:

```python
from naive_bayes import MixedNB
```

Then use the model in the following way.
When initializing the model specify which features are categorical, and which are numerical, with the argument `categorical_feature_mask`:

```python
model = MixedNB(categorical_feature_mask=[True,False,False,True])
```

Then you can simply use the model like any other scikit-learn classifier.
First train the model with `fit`, and later use `predict` to use the trained model:

```python
model.fit(X,y)
y_pred = model.predict(X_test)
```

Additionally, you can run the jupyter notebooks `naive_bayes_comparison.ipynb` and `naive_bayes_testcases.ipynb`.

For more details, you can look at the jupyter notebooks or at the comments/docstrings in the source code.

## Implementation

- Handle categorical and numeric data at the same time!
  - Specify datatype at initialization
- Sklearn-style interface
    - For interoperability with sklearn pipelines etc.
- Fitting:
  - Categorical:
    - Count relative frequencies
    - Laplace Smoothing
  - Numeric:
    - Calculate gaussian distribution (mean, variance)
    - Variance Smoothing (for numeric stability, inspired by scikit-learn)
- Predicting:
  - Use Log-Likelihoods when predicting


## Evaluation

Compare the performance of our approach against other algorithms.

We use 2 datasets that are quite different from each other:

| Dataset             | Type of data | Observations | Features | Target Classes |
|---------------------|--------------|-------------:|---------:|---------------:|
| Amazon Review       | numeric      |          750 |    10000 |             50 |
| Car Insurance Claim | categorical  |        10000 |       12 |              2 |


We compare *Our Naive Bayes* against *Scikit-learns Naive Bayes* and *Scikit-learns Random Forrest*:

| Dataset             | Model                       | Accuracy | Fitting time (s) | Score time (s) |
|---------------------|-----------------------------|----------|------------------|----------------|
| Amazon Review       | Our Naive Bayes*            | 0.65     | 0.588            | 1.884          |
| --//--              | Scikit-learn Naive Bayes*   | 0.65     | 0.588            | 0.188          |
| --//--              | Scikit-learn Random Forrest | 0.61     | 0.551            | 0.011          |
| Car Insurance Claim | Our Naive Bayes**           | 0.80     | 0.064            | 0.900          |
| --//--              | Scikit-learn Naive Bayes**  | 0.80     | 0.006            | 0.005          |
| --//--              | Scikit-learn Random Forrest | 0.81     | 0.777            | 0.037          |

*Laplace-Smoothing=50
**Variance-Smoothing=0.01

Our implementation reaches similar accuracy as scikit-learns implementation.
However, the fitting (aka training) time and the scoring (aka testing) time are substantially slower.

For more details look at the jupyter notebook `naive_bayes_comparison.ipynb`.

## File Overview

Source Code (`/src`):

- `naive_bayes.py`: Mixed Naive Bayes Algorithm
- `naive_bayes_testcases.ipynb`: Testcases to check if our naive-bayes implementation works as expected
- `naive_bayes_comparison.ipynb`: Compare our algorithm to other approaches and do hyperparameter tuning
- `utils.py:` contains utils e.g. function for plotting the hyperparameters of gridsearch, function for plotting results as a table

Datasets (`/data`):

- `amazon_review.csv`: Amazon Reviews, adapted version of the dataset published on [UCI Machine Learning Repository](https://doi.org/10.24432/C55C88)
- `car_insurance_claims.csv`: Car Insurance Claims, published on [Kaggle](https://www.kaggle.com/datasets/sagnik1511/car-insurance-data)
- `naive-bayes_example_cat+num.csv`: Very simple Weather dataset, with categorical & numeric attributes
- `naive-bayes_example_cat.csv`: Very simple Weather dataset, with categorical attributes


## License

The source code in `/src` is licensed under the [MIT license](/src/LICENSE).
For the datasets in `/data` best check the license of the original source.