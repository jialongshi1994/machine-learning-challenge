# data manipulation
import pandas as pd
import numpy as np

# visualization
import seaborn as sb
import matplotlib.pyplot as plt
# %matplotlib inline

# model training
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# model evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# classifiers
from sklearn.naive_bayes import GaussianNB # naive bayes
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.linear_model import LogisticRegression # logistic regression
from sklearn.tree import DecisionTreeClassifier # decision Tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("exoplanet_data.csv")
# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')
# Drop the null rows
df = df.dropna()

# Scale your data
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif, chi2

x = df.drop(columns='koi_disposition')
y = df['koi_disposition']
train = SelectKBest(score_func=f_classif, k=5)
train.fit(x, y)
selected_features = x.iloc[:, train.get_support(True)]
selected_features.head()
X_train, X_test, y_train, y_test = train_test_split(selected_features, y, train_size=0.8, test_size=0.2, random_state=200)

X_tr = X_train[:]
X_te = X_test[:]
min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_tr)
X_test_scaled = min_max_scaler.fit_transform(X_te)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
X_train_scaled.head()
X_test_scaled.head()


def getAccF1(model, dataset):
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(y_pred)
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    f1 = round(f1_score(y_test, y_pred, average='weighted') * 100, 2)
    return acc, f1


if __name__ == "__main__":
    # K-nearest neighbors, Logistic regression, Decision trees, Random forest, Gradient boosting machine
    model_names = ['KNN', 'LR', 'DT', 'RF', 'GBM']
    Acc = []
    F1 = []

    # --- Your code here ---
    knn = KNeighborsClassifier()
    lr = LogisticRegression()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    gbm = GradientBoostingClassifier()
    model = [knn, lr, dt, rf, gbm]
    # please use the following function to calculate f1 and acc
    # f1 = round(f1_score(y_test, y_pred, average='weighted') * 100, 2)
    # acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    for i in range(len(model)):
        a1, f1 = getAccF1(model[i], 0)
        Acc.append(a1)
        F1.append(f1)
    # --- End of your code ---

    Record = pd.DataFrame({'Model': model_names, 'acc': Acc, 'f1': F1})
    Record['acc_mean'] = Record.mean(axis=1).round(2)
    Record['F1_mean'] = Record.mean(axis=2).round(2)
    Record.set_index('Model', inplace=True)
    Record.loc['avg'] = Record.mean()

    print(Record)

