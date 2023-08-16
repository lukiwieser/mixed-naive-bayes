# Mixed Naive Bayes

Naive Bayes Classifier for categorical and/or numeric features.

Scikit learn currently only supports numeric data (with GaussianNB) or categorical data (with CategoricalNB), but not both. 
That's why I created this prototype.

This project uses Python 3.9.

Source Code (`/src`):

- `naive_bayes.py`: Mixed Naive Bayes Algorithm
- `naive_bayes_testcases`: Testcases to check if our naive-bayes implementation works as expected
- `naive_bayes_comparison`: Compare our algorithm to other approaches and do hyperparameter tuning
- `utils.py:` contains utils e.g. function for plotting the hyperparameters of gridsearch, function for plotting results as a table

Datasets (`/data`):

- `amazon_review.csv`: Amazon Reviews, adapted version of the dataset published on [UCI Machine Learning Repository](https://doi.org/10.24432/C55C88)
- `car_insurance_claims.csv`: Car Insurance Claims, published on [Kaggle](https://www.kaggle.com/datasets/sagnik1511/car-insurance-data)
- `naive-bayes_example_cat+num.csv`: Very simple Weather dataset, with categorical & numeric attributes
- `naive-bayes_example_cat.csv`: Very simple Weather dataset, with categorical attributes
