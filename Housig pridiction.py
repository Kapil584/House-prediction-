import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.metrics import *
import joblib



df = pd.read_csv("Housing.csv")
df = df.drop_duplicates()
le = LabelEncoder()
binary_cols = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea'
]
for col in binary_cols:
    df[col] = le.fit_transform(df[col])


df = pd.get_dummies(
    df,
    columns=['furnishingstatus'],
    drop_first=True
)



print(df.head())
print(df.info())
X = df.drop('price', axis=1)

y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("MAE:", mae)
mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)
r2 = r2_score(y_test, y_pred)

print("R2 Score:", r2)
binary_cols = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea'
]


    
print(df.head(12))   
joblib.dump(model, "housing_model.pkl")
joblib.dump(X.columns, "columns.pkl") 