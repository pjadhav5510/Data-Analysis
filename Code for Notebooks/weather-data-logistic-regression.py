# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T04:00:21.066528Z","iopub.execute_input":"2024-03-01T04:00:21.066977Z","iopub.status.idle":"2024-03-01T04:00:21.116071Z","shell.execute_reply.started":"2024-03-01T04:00:21.066938Z","shell.execute_reply":"2024-03-01T04:00:21.115028Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T04:00:21.118249Z","iopub.execute_input":"2024-03-01T04:00:21.118865Z","iopub.status.idle":"2024-03-01T04:00:21.123809Z","shell.execute_reply.started":"2024-03-01T04:00:21.118827Z","shell.execute_reply":"2024-03-01T04:00:21.123000Z"}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T04:00:21.125168Z","iopub.execute_input":"2024-03-01T04:00:21.125483Z","iopub.status.idle":"2024-03-01T04:00:21.922055Z","shell.execute_reply.started":"2024-03-01T04:00:21.125456Z","shell.execute_reply":"2024-03-01T04:00:21.920935Z"}}
df = pd.read_csv("/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv")
df

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T04:00:21.923362Z","iopub.execute_input":"2024-03-01T04:00:21.923717Z","iopub.status.idle":"2024-03-01T04:00:22.269208Z","shell.execute_reply.started":"2024-03-01T04:00:21.923687Z","shell.execute_reply":"2024-03-01T04:00:22.268003Z"}}
df.isnull().sum()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T04:00:22.272767Z","iopub.execute_input":"2024-03-01T04:00:22.273780Z","iopub.status.idle":"2024-03-01T04:00:22.302523Z","shell.execute_reply.started":"2024-03-01T04:00:22.273734Z","shell.execute_reply":"2024-03-01T04:00:22.301106Z"}}
df.drop(['WindDir9am'],axis=1,inplace=True)
df.columns

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T04:00:22.304285Z","iopub.execute_input":"2024-03-01T04:00:22.305108Z","iopub.status.idle":"2024-03-01T04:00:22.654179Z","shell.execute_reply.started":"2024-03-01T04:00:22.305059Z","shell.execute_reply":"2024-03-01T04:00:22.653003Z"}}
df.dropna(axis=0,inplace=True)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T04:00:22.655532Z","iopub.execute_input":"2024-03-01T04:00:22.655947Z","iopub.status.idle":"2024-03-01T04:00:22.783811Z","shell.execute_reply.started":"2024-03-01T04:00:22.655911Z","shell.execute_reply":"2024-03-01T04:00:22.782682Z"}}
df.isnull().sum()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T04:00:22.785606Z","iopub.execute_input":"2024-03-01T04:00:22.786419Z","iopub.status.idle":"2024-03-01T04:00:22.827992Z","shell.execute_reply.started":"2024-03-01T04:00:22.786383Z","shell.execute_reply":"2024-03-01T04:00:22.826801Z"}}
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T04:00:22.829350Z","iopub.execute_input":"2024-03-01T04:00:22.829790Z","iopub.status.idle":"2024-03-01T04:00:22.983150Z","shell.execute_reply.started":"2024-03-01T04:00:22.829748Z","shell.execute_reply":"2024-03-01T04:00:22.981841Z"}}
df.info()

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T04:00:22.985239Z","iopub.execute_input":"2024-03-01T04:00:22.986142Z","iopub.status.idle":"2024-03-01T04:00:25.753181Z","shell.execute_reply.started":"2024-03-01T04:00:22.986091Z","shell.execute_reply":"2024-03-01T04:00:25.751145Z"}}
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = df.drop(['RainTomorrow','Date','Location','WindGustDir','WindDir3pm','RainToday'],axis=1)
#.columns
y = df['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
#clf = LogisticRegression()
model.fit(X_train, y_train)

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T04:00:25.755583Z","iopub.execute_input":"2024-03-01T04:00:25.759712Z","iopub.status.idle":"2024-03-01T04:00:25.819434Z","shell.execute_reply.started":"2024-03-01T04:00:25.759650Z","shell.execute_reply":"2024-03-01T04:00:25.818302Z"}}
y_pred = model.predict(X_test)
y_pred,X_test

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T04:00:25.821667Z","iopub.execute_input":"2024-03-01T04:00:25.823235Z","iopub.status.idle":"2024-03-01T04:00:25.925194Z","shell.execute_reply.started":"2024-03-01T04:00:25.823194Z","shell.execute_reply":"2024-03-01T04:00:25.923867Z"}}
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T04:00:25.927032Z","iopub.execute_input":"2024-03-01T04:00:25.927735Z","iopub.status.idle":"2024-03-01T04:00:26.114329Z","shell.execute_reply.started":"2024-03-01T04:00:25.927691Z","shell.execute_reply":"2024-03-01T04:00:26.113148Z"}}
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T04:00:26.118936Z","iopub.execute_input":"2024-03-01T04:00:26.119637Z","iopub.status.idle":"2024-03-01T04:00:27.116899Z","shell.execute_reply.started":"2024-03-01T04:00:26.119601Z","shell.execute_reply":"2024-03-01T04:00:27.115526Z"}}
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T04:00:27.118947Z","iopub.execute_input":"2024-03-01T04:00:27.119695Z","iopub.status.idle":"2024-03-01T04:00:27.240024Z","shell.execute_reply.started":"2024-03-01T04:00:27.119652Z","shell.execute_reply":"2024-03-01T04:00:27.239104Z"}}
cm = confusion_matrix(y_test, y_pred)
TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

# %% [code] {"execution":{"iopub.status.busy":"2024-03-01T04:00:27.241417Z","iopub.execute_input":"2024-03-01T04:00:27.243824Z","iopub.status.idle":"2024-03-01T04:00:27.250410Z","shell.execute_reply.started":"2024-03-01T04:00:27.243787Z","shell.execute_reply":"2024-03-01T04:00:27.249011Z"}}
precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))
