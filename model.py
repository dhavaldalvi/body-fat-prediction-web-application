import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

# Loading dataset
df = pd.read_csv("data/bodyfat.csv")

# Height and weight is swapped
df['Height'], df['Weight'] = df['Weight'], df['Height']
df.head()

# Dropping Density, because the are related by the equation given in the description
data = df.drop(['Density'], axis = True)

# Function to remove outliers
def remove_outliers(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 1.5*iqr
    lower_limit = q1 - 1.5*iqr
    return df[~((df < lower_limit) | (df > upper_limit)).any(axis=1)]

# Removing Outliers
data = remove_outliers(data)

# Selecting features and target
X = data.drop(['BodyFat'], axis = 1)
y = data['BodyFat']

# Splitting data into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 45)

# fitting the scaler
# scaler = StandardScaler()
# scaler.fit(X_train)

# Scaling the features in training and testing dataset
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Initialize and train the Lasso Regression model
# model = Lasso(alpha = np.log(1.19))
# model.fit(X_train_scaled, y_train)

# Fitting and Training
pipeline = make_pipeline(StandardScaler(), Lasso(alpha=0.75))
pipeline.fit(X_train, y_train)

# Saving model
pickle.dump(pipeline, open('model.pkl', 'wb'))