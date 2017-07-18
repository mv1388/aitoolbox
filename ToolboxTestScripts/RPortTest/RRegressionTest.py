from AIToolbox.RPort.RRegressionModels import RRegressionModel
from sklearn import datasets

from numpy.random import randint


iris_ds = datasets.load_iris()
X = iris_ds.data[:120, :-1]
# y = iris_ds.data[:120, -1]
y = randint(1, 50, 120)

X_test = iris_ds.data[120:, :-1]
y_test = iris_ds.data[120:, -1]


# Poisson regression in R
# summary(m1 <- glm(num_awards ~ prog + math, family="poisson", data=p))

rrm = RRegressionModel(X, y)
prediction = rrm.poisson_regression_fit("y_attr ~ .", "poisson").predict(X_test)

print y
print prediction
