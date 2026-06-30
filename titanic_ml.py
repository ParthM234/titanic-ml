import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv("train.csv")

# ---- Data Loading and Cleaning ----

df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"],inplace=True)
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

#Categories to numbers
df["Sex"]=df["Sex"].map({"male":0, "female":1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

#---- Feature Engineering ----
X= df.drop(columns=["Survived"])
y= df["Survived"]

# ---- Train Test Split ----
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

print("Data ready")
print("Features:", X.columns.tolist())
print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ---- Model Training ----

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))
print("Logistic Regression Accuracy:", lr_acc)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test))
print("KNN Accuracy:", knn_acc)

svm = SVC()
svm.fit(X_train, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test))
print("SVM Accuracy:", svm_acc)

# ---- Model Comparison ----
models = {"Logistic Regression": lr_acc, "KNN": knn_acc, "SVM": svm_acc}
best_model = max(models, key=models.get)
print("Best Model:", best_model)

# ---- Visualization ----
from matplotlib import pyplot as plt
model_names=list(models.keys())
accuracies=list(models.values())

plt.bar(model_names, accuracies, color=["blue", "orange", "green"])
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0.5, 1.0)
for i, acc in enumerate(accuracies):
    plt.text(i, acc+0.01, f"{acc:.2f}",  ha="center")
plt.savefig("model_comparison.png")
plt.show()

#Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm= confusion_matrix(y_test, lr.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

