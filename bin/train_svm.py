from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("/opt/final_vectors.csv", header=None)
df = abs(new_df)
df = df.rename(columns={301:'target'})

## Split the data for training
x_train, x_test, y_train, y_test = train_test_split(df.drop(['target'], axis=1), df.target, test_size=0.33)

## Train SVM
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

print("test score: ", accuracy_score(y_pred, y_test))

