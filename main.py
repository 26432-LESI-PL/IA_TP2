import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, classification_report, confusion_matrix, root_mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset
df = pd.read_csv("housing.txt")

# --- Data Exploration ---

# 1. Descriptive Statistics and Visualization
print(df.describe())

# Histograms for numerical features
df.hist(bins=50, figsize=(20, 15))
plt.suptitle("Histograms of Numerical Features", y=1.02)
plt.show()

# Box plots for numerical features
plt.figure(figsize=(20, 15))
for i, col in enumerate(df.select_dtypes(include=['float64', 'int64']).columns):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.suptitle("Box Plots of Numerical Features", y=1.02)
plt.show()

# Count plot for the categorical feature
plt.figure(figsize=(8, 6))
sns.countplot(x='ocean_proximity', data=df)
plt.title("Count of Ocean Proximity Categories")
plt.show()

# --- Data Preprocessing ---

# 3. Feature Engineering
df['rooms_per_household'] = df['total_rooms'] / df['households']

# OPTION 1: Remove rows with NaNs (if applicable)
df.dropna(inplace=True)

# OPTION 2: Impute missing values (using SimpleImputer)
# numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
# categorical_features = ['ocean_proximity']

# Create a column transformer for preprocessing
""" preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]) """

# Apply preprocessing to training and testing data
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Impute missing values in numerical features using the mean
""" imputer = SimpleImputer(strategy='mean')
X[numerical_features] = imputer.fit_transform(X[numerical_features]) """

""" X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test) """

# 4. Data Splitting
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Feature Scaling and One-Hot Encoding
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = ['ocean_proximity']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Apply preprocessing to training and testing data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# --- Model Building and Evaluation ---

# 6. Model Training (Linear Regression, Decision Tree, Random Forest, HistGradientBoostingRegressor)
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 7. Model Evaluation (RMSE, R-squared)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model: {name}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2}")
    print("-" * 30)

# --- Clustering (K-Means) ---
# 8. Determine optimal k using the Elbow method and Silhouette analysis
inertia = []
silhouette = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train)
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X_train, kmeans.labels_))

# Plot Elbow Method
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')

# Plot Silhouette Analysis
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette, marker='o')
plt.title('Silhouette Analysis for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()

# 9. Train K-Means with chosen k (e.g., k=5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_train)
cluster_labels = kmeans.labels_

# --- Association Rule Mining (Apriori) ---

# 10. Discretize features for Apriori (example: using qcut for numerical features)
X_apriori = X.copy()
for col in numerical_features:
    X_apriori[col] = pd.qcut(X_apriori[col], q=4, duplicates='drop')

# Convert categorical features to string type before one-hot encoding
for col in categorical_features:
    X_apriori[col] = X_apriori[col].astype(str)

# Convert to one-hot format for Apriori
X_apriori = pd.get_dummies(X_apriori)

# 11. Apply Apriori algorithm
frequent_itemsets = apriori(X_apriori, min_support=0.1, use_colnames=True)

# 12. Generate association rules
num_transactions = len(df)  # Total rows in the original DataFrame
rules = association_rules(frequent_itemsets, num_transactions, metric="confidence", min_threshold=0.7)

# Display frequent itemsets and association rules
print("Frequent Itemsets:")
print(frequent_itemsets.head())
print("\nAssociation Rules:")
print(rules.head())

# --- Classification (example using Random Forest Classifier) ---

# 13. Prepare data for classification (using 'ocean_proximity' as target)
X_class = df.drop('ocean_proximity', axis=1)
y_class = df['ocean_proximity']

# Encode the target variable for classification
label_encoder = LabelEncoder()
y_class_encoded = label_encoder.fit_transform(y_class)

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class_encoded, test_size=0.2, random_state=42)

# Preprocess numerical features for classification
preprocessor_class = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features)
    ])

X_train_class = preprocessor_class.fit_transform(X_train_class)
X_test_class = preprocessor_class.transform(X_test_class)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_class, y_train_class)

# Predict and evaluate
y_pred_class = rf_classifier.predict(X_test_class)
print("\nClassification Report:")
print(classification_report(y_test_class, y_pred_class))
print("Confusion Matrix:")
print(confusion_matrix(y_test_class, y_pred_class))