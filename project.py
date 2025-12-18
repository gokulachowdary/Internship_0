# ============================================
# Tasks 1-5 Preprocessing, Split, Feature Selection, Model selection, modelTraining
# ============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading the dataset
df = pd.read_csv("dataset.csv")


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# handling or Fill missing values
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Fill categorical missing values with mode (just in case)
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 4 Encode target variable
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

#  Choose features we think are important
features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 
            'PaymentMethod', 'InternetService', 'OnlineSecurity', 
            'TechSupport', 'SeniorCitizen', 'Partner', 'Dependents', 
            'PaperlessBilling']

# 6 Encode categorical features
categorical_features = ['Contract', 'PaymentMethod', 'InternetService', 
                        'OnlineSecurity', 'TechSupport', 'Partner', 
                        'Dependents', 'PaperlessBilling']

df = pd.get_dummies(df[features + ['Churn']], columns=categorical_features, drop_first=True)


X = df.drop('Churn', axis=1)
y = df['Churn']

# 8 Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100)
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n=============================")
    print(f"Model: {model_name}")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
