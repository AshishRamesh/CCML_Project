import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import datetime

# Load the exported CSV
df = pd.read_csv("tasks_dataset.csv")

# Convert date strings back to datetime
df['created_at'] = pd.to_datetime(df['created_at'])
df['due_date'] = pd.to_datetime(df['due_date'])

# Feature Engineering
df['days_until_due'] = (df['due_date'] - df['created_at']).dt.days
df['hour_created'] = df['created_at'].dt.hour
df['day_of_week_created'] = df['created_at'].dt.dayofweek

# Encode priority into numerical values
priority_encoder = LabelEncoder()
df['priority_encoded'] = priority_encoder.fit_transform(df['priority'])

# Define features and target
X = df[['days_until_due', 'hour_created', 'day_of_week_created', 'priority_encoded']]
y = df['completed'].astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# (Optional) Save the model
import joblib
joblib.dump(clf, 'task_completion_model.pkl')
