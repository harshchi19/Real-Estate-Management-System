import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

# Load the dataset
df = pd.read_csv('mumbai.csv')

# Drop rows with missing target value (PRICE_PER_UNIT_AREA) and split into features and target
df_clean = df.dropna(subset=['PRICE_PER_UNIT_AREA'])
X = df_clean.drop(columns=['PRICE_PER_UNIT_AREA'])
y = df_clean['PRICE_PER_UNIT_AREA']

# Convert categorical columns to category datatype
categorical_cols = X.select_dtypes(include=['object']).columns
X[categorical_cols] = X[categorical_cols].astype('category')

# Get the indices of categorical columns
cat_feature_indices = [X.columns.get_loc(col) for col in categorical_cols]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM Model
lgb_model = lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.1)

# Train the LightGBM model
lgb_model.fit(X_train, y_train)

# Predict and evaluate LightGBM model
lgb_pred = lgb_model.predict(X_val)
lgb_mae = mean_absolute_error(y_val, lgb_pred)
lgb_r2 = r2_score(y_val, lgb_pred)

print(f'LightGBM Model MAE: {lgb_mae}')
print(f'LightGBM Model R² Score: {lgb_r2}')

# CatBoost Model (pass categorical feature indices)
cat_model = CatBoostRegressor(random_seed=42, iterations=1000, depth=6, learning_rate=0.1, verbose=False)

# Train the CatBoost model
cat_model.fit(X_train, y_train, cat_features=cat_feature_indices)

# Predict and evaluate CatBoost model
cat_pred = cat_model.predict(X_val)
cat_mae = mean_absolute_error(y_val, cat_pred)
cat_r2 = r2_score(y_val, cat_pred)

print(f'CatBoost Model MAE: {cat_mae}')
print(f'CatBoost Model R² Score: {cat_r2}')
