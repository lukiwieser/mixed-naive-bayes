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

For more details you can look at the jupyter notebooks or at the comments/docstrings in the source code.


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
