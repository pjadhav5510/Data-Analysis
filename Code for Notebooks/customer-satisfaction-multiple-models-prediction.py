# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T03:44:50.964890Z","iopub.execute_input":"2024-03-01T03:44:50.965281Z","iopub.status.idle":"2024-03-01T03:44:54.117507Z","shell.execute_reply.started":"2024-03-01T03:44:50.965249Z","shell.execute_reply":"2024-03-01T03:44:54.116280Z"}}

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# # Load df

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T03:44:54.119700Z","iopub.execute_input":"2024-03-01T03:44:54.120477Z","iopub.status.idle":"2024-03-01T03:44:54.518629Z","shell.execute_reply.started":"2024-03-01T03:44:54.120439Z","shell.execute_reply":"2024-03-01T03:44:54.517162Z"}}
df = pd.read_csv('/kaggle/input/customer-satisfaction-in-airline/Invistico_Airline.csv')
df.sample(30).T

# %% [markdown]
# # Check for nan values

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T03:44:54.519809Z","iopub.execute_input":"2024-03-01T03:44:54.520146Z","iopub.status.idle":"2024-03-01T03:44:54.547960Z","shell.execute_reply.started":"2024-03-01T03:44:54.520117Z","shell.execute_reply":"2024-03-01T03:44:54.546405Z"}}
df.isnull().sum()

# %% [markdown]
# # Imput nan values

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T03:44:54.550351Z","iopub.execute_input":"2024-03-01T03:44:54.550696Z","iopub.status.idle":"2024-03-01T03:44:54.559207Z","shell.execute_reply.started":"2024-03-01T03:44:54.550668Z","shell.execute_reply":"2024-03-01T03:44:54.557944Z"}}
df['Arrival Delay in Minutes']

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T03:44:54.560915Z","iopub.execute_input":"2024-03-01T03:44:54.561313Z","iopub.status.idle":"2024-03-01T03:44:54.577795Z","shell.execute_reply.started":"2024-03-01T03:44:54.561282Z","shell.execute_reply":"2024-03-01T03:44:54.577083Z"}}
mean = df['Arrival Delay in Minutes'].mean()
mean

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T03:44:54.578823Z","iopub.execute_input":"2024-03-01T03:44:54.579310Z","iopub.status.idle":"2024-03-01T03:44:54.588219Z","shell.execute_reply.started":"2024-03-01T03:44:54.579284Z","shell.execute_reply":"2024-03-01T03:44:54.586665Z"}}
df['Arrival Delay in Minutes'].replace(np.nan, mean, inplace=True)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T03:44:54.589667Z","iopub.execute_input":"2024-03-01T03:44:54.590099Z","iopub.status.idle":"2024-03-01T03:44:54.620582Z","shell.execute_reply.started":"2024-03-01T03:44:54.590063Z","shell.execute_reply":"2024-03-01T03:44:54.619627Z"}}
df.isnull().sum()

# %% [markdown]
# # Label Encoder for categorical columns.

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T03:44:54.622187Z","iopub.execute_input":"2024-03-01T03:44:54.622754Z","iopub.status.idle":"2024-03-01T03:44:54.632954Z","shell.execute_reply.started":"2024-03-01T03:44:54.622720Z","shell.execute_reply":"2024-03-01T03:44:54.631260Z"}}
df.dtypes

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T03:44:54.634601Z","iopub.execute_input":"2024-03-01T03:44:54.635842Z","iopub.status.idle":"2024-03-01T03:44:54.747938Z","shell.execute_reply.started":"2024-03-01T03:44:54.635808Z","shell.execute_reply":"2024-03-01T03:44:54.746857Z"}}
le = LabelEncoder()
df['satisfaction'] = le.fit_transform(df['satisfaction'])
df['Customer Type'] = le.fit_transform(df['Customer Type'])
df['Class'] = le.fit_transform(df['Class'])
df['Type of Travel'] = le.fit_transform(df['Type of Travel'])

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T03:44:54.750694Z","iopub.execute_input":"2024-03-01T03:44:54.751820Z","iopub.status.idle":"2024-03-01T03:44:54.770126Z","shell.execute_reply.started":"2024-03-01T03:44:54.751781Z","shell.execute_reply":"2024-03-01T03:44:54.769052Z"}}
df.head()

# %% [markdown]
# # Split data & predictions with multiple classifiers.

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T03:44:54.771456Z","iopub.execute_input":"2024-03-01T03:44:54.772180Z","iopub.status.idle":"2024-03-01T03:45:59.358989Z","shell.execute_reply.started":"2024-03-01T03:44:54.772144Z","shell.execute_reply":"2024-03-01T03:45:59.358229Z"}}
X = df.drop('satisfaction', axis=1)
y = df['satisfaction']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

models = [
    CatBoostClassifier(verbose=False),
    XGBClassifier(),
    GaussianNB(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    LogisticRegression(max_iter=2000)
    
]

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_pred, y_test)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    model_name = model.__class__.__name__
    print(f'{model_name} - Precisión: {pre:.2f}')
    print(f'{model_name} - Accuracy: {score:.2f}')
    print(f'{model_name} - Recall: {rec:.2f}')
    print(f'{model_name} - F1 Score: {f1:.2f}')


# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T03:45:59.362552Z","iopub.execute_input":"2024-03-01T03:45:59.364525Z","iopub.status.idle":"2024-03-01T03:47:03.545679Z","shell.execute_reply.started":"2024-03-01T03:45:59.364491Z","shell.execute_reply":"2024-03-01T03:47:03.543996Z"}}
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    con = confusion_matrix(y_test, y_pred)
    model_name = model.__class__.__name__
    sns.heatmap(con, annot=True, fmt='d')
    plt.show()
