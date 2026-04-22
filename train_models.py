import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("data/NIJ_s_Recidivism_Challenge_Full_Dataset_20240402.csv")
df = df.iloc[:, 1:]

cols_to_drop = [
    "Recidivism_Arrest_Year1",
    "Recidivism_Arrest_Year2",
    "Recidivism_Arrest_Year3"
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

target = "Recidivism_Within_3years"
X = df.drop(columns=[target])
y = df[target]

categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns
numeric_cols = X.select_dtypes(include=["number"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=1),
    "gradient_boosting": GradientBoostingClassifier(random_state=1),
    "svm": SVC(probability=True, random_state=1),
    "neural_network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=1)
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1
)

results = []

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_test)[:, 1]
        row["roc_auc"] = roc_auc_score(y_test, probs)

    results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv("outputs/model_metrics.csv", index=False)
print(results_df.sort_values("f1", ascending=False))