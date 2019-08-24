import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("/opt/data/final_vectors.csv", header=None, names=['doc_len_diff', 'doc_len_diff_pct', 'word_share_pct', 'euc_dist', 'target'])

## Standardize the Values
for col in df.columns:
    if col == 'target':
        continue
    df[col] = StandardScaler().fit_transform(np.reshape(df[col].values, (-1,1)))

## Split the data for training
x_train, x_test, y_train, y_test = train_test_split(df.drop(['target'], axis=1), df.target, test_size=0.33, shuffle=True)

## Train logistic regression
lr = LogisticRegression(penalty='l2', fit_intercept=True)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print("test score: ", accuracy_score(y_pred, y_test))

