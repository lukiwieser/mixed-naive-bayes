# Mixed Naive Bayes

Naive Bayes Classifier for categorical and/or numeric features.

Motivation:
Scikit-learn currently only supports numeric features (with GaussianNB) or categorical features (with CategoricalNB), but not both at the same time.


## Contents

- [Usage](#usage)
- [Files](#files)
- [Implementation](#implementation)
- [Evaluation](#evaluation)
- [Lessons Learned](#lessons-learned)
- [License](#license)


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
When initializing the model, specify which features are categorical and which are numerical, with the argument `categorical_feature_mask`:

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

For more details, you can look at the jupyter notebooks or the comments/docstrings in the source code.


## Files

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


## Implementation

- Handle categorical and numeric data at the same time!
  - Specify datatype at initialization
- Sklearn-style interface
    - For interoperability with sklearn pipelines etc.
- Training (Fitting):
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

| Dataset             | Type of Data | Observations | Features | Target Classes |
|---------------------|--------------|-------------:|---------:|---------------:|
| Amazon Review       | numeric      |          750 |    10000 |             50 |
| Car Insurance Claim | categorical  |        10000 |       12 |              2 |


We compare *Our Naive Bayes* against *Scikit-learns Naive Bayes* and *Scikit-learns Random Forest*:

| Dataset             | Model                      | Accuracy | Fit Time (ms) | Score Time (ms) |
|---------------------|----------------------------|---------:|--------------:|----------------:|
| Amazon Review       | Our Naive Bayes*           |     0.65 |           588 |            1884 |
| --//--              | Scikit-learn Naive Bayes*  |     0.65 |           588 |             188 |
| --//--              | Scikit-learn Random Forest |     0.61 |           551 |              11 |
| Car Insurance Claim | Our Naive Bayes**          |     0.80 |            64 |             900 |
| --//--              | Scikit-learn Naive Bayes** |     0.80 |             6 |               5 |
| --//--              | Scikit-learn Random Forest |     0.81 |           777 |              37 |

*Laplace-Smoothing=50
**Variance-Smoothing=0.01

Our implementation reaches similar accuracy as scikit-learns implementation.
However, the fit (aka training) time and the score (aka testing) time are substantially slower.

Random Forest has the highest fit time, but makes up for it with a quite fast score time.

For more details look at the jupyter notebook `naive_bayes_comparison.ipynb`.


## Lessons Learned

- Our implementation is very effective, but slowish.
- Take numerical stability into account e.g.:
  - Log-likelihoods (when multiplying lots of small probabilities together)
  - Variance-smoothing (so that variance values do not get too small)
- Efficiently working with pandas/numpy is important
  - Can turn hours into seconds!  
  - e.g. by using bulk operations instead of lots of loops
- Garbage in, garbage out
  - Accidentally doing nonsense can happen more easily than expected
  - e.g. do one-hot-encoding, then apply gaussian naive bayes
- Input validation can avoid foolish mistakes
- It is easy to make mistakes, the algorithm can run but still output wrong calculations


## License

The source code in `/src` is licensed under the [MIT license](/src/LICENSE).
For the datasets in `/data` best check the license of the original source.
