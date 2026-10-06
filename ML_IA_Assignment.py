
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, silhouette_score

from xgboost import XGBRegressor


# =========================
# 1. BOOSTING
# =========================

insurance = pd.read_csv("insurance_pre.csv")

X = insurance.drop("charges", axis=1)
y = insurance["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker"]),
        ("num", "passthrough", ["age", "bmi", "children"])
    ]
)

boosting_models = {
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
}

boosting_results = []

for name, model in boosting_models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    boosting_results.append([name, mae, rmse, r2])

boosting_df = pd.DataFrame(
    boosting_results,
    columns=["Algorithm", "MAE", "RMSE", "R2 Score"]
)

print("\nBOOSTING RESULTS")
print(boosting_df)


# =========================
# 2. CLASSIFICATION
# =========================

social = pd.read_csv("Social_Network_Ads.csv")

X = social.drop(["Purchased", "User ID"], axis=1)
y = social["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Gender"]),
        ("num", StandardScaler(), ["Age", "EstimatedSalary"])
    ]
)

classification_models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

classification_results = []

for name, model in classification_models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    classification_results.append([name, accuracy])

classification_df = pd.DataFrame(
    classification_results,
    columns=["Algorithm", "Accuracy"]
)

print("\nCLASSIFICATION RESULTS")
print(classification_df)


# =========================
# 3. CLUSTERING
# =========================

mall = pd.read_csv("Mall_Customers.csv")

X_cluster = mall[
    ["Annual Income (k$)", "Spending Score (1-100)"]
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# K-Means
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_score = silhouette_score(X_scaled, kmeans_labels)


# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=5)

hierarchical_labels = hierarchical.fit_predict(X_scaled)
hierarchical_score = silhouette_score(
    X_scaled,
    hierarchical_labels
)


# DBSCAN
dbscan = DBSCAN(
    eps=0.5,
    min_samples=5
)

dbscan_labels = dbscan.fit_predict(X_scaled)

unique_labels = set(dbscan_labels)

if len(unique_labels - {-1}) > 1:
    dbscan_score = silhouette_score(
        X_scaled,
        dbscan_labels
    )
else:
    dbscan_score = -1


clustering_df = pd.DataFrame({
    "Algorithm": [
        "K-Means",
        "Hierarchical Clustering",
        "DBSCAN"
    ],
    "Silhouette Score": [
        kmeans_score,
        hierarchical_score,
        dbscan_score
    ]
})

print("\nCLUSTERING RESULTS")
print(clustering_df)


# =========================
# FINAL RESULTS
# =========================

best_boosting = boosting_df.loc[
    boosting_df["MAE"].idxmin()
]

best_classification = classification_df.loc[
    classification_df["Accuracy"].idxmax()
]

best_clustering = clustering_df.loc[
    clustering_df["Silhouette Score"].idxmax()
]

final_results = pd.DataFrame({
    "Category": [
        "Boosting",
        "Classification",
        "Clustering"
    ],
    "Best Algorithm": [
        best_boosting["Algorithm"],
        best_classification["Algorithm"],
        best_clustering["Algorithm"]
    ],
    "Best Result": [
        best_boosting["R2 Score"],
        best_classification["Accuracy"],
        best_clustering["Silhouette Score"]
    ]
})

print("\nFINAL RESULT TABLE")
print(final_results)
