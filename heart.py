import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression 
import seaborn as sns
#from sklearn.metrics import predict_proba
import pickle 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split



df=pd.read_csv('D:/Streamlit/heart.csv')

pd.options.display.max_columns=50
df.head()
df.describe()
df['RestingBP'].mean()
df['RestingBP']=df['RestingBP'].replace(0, df['RestingBP'].mean())
df['RestingBP'].min()
df['RestingBP'].max()


df.describe()
df['FastingBS']=df['FastingBS'].replace(0, df['FastingBS'].mean())
df.describe()
df.head(50)
df['Age'].max()

### Converted the Age Column into a bin range using pd.cut fx
category=pd.cut(df.Age, bins=[25,35,45,55,65,80], labels=['25-35','35-45','45-55','55-65','65-80'])

df.insert(2,'AgeGroup', category)
df.head()


le=LabelEncoder()
### Encoding the Various Column

df['Sex']=le.fit_transform(df.Sex)
df['AgeGroup']=le.fit_transform(df['AgeGroup'])
df['ExerciseAngina']=le.fit_transform(df.ExerciseAngina)
df['ChestPainType']=le.fit_transform(df.ChestPainType)
df['ST_Slope']=le.fit_transform(df['ST_Slope'])
df['RestingECG']=le.fit_transform(df.RestingECG)

df.head()
### Fitting the correlation chart
df.corr()
## Visualizing the Correlation
sns.heatmap(df.corr())
df
list(le.classes_)
df.head()
df.Age.max()
x_train=df.iloc[:,1:12]
x_train
y_train=df['HeartDisease']
y_train
X_train, X_test, y_train, y_test=train_test_split(x_train, y_train, test_size=30, random_state=100)

X_train 
y_train 
LOGREG=LogisticRegression()

model=LOGREG.fit(X_train, y_train)
print(x_train[9:10])
y_pred=model.predict(X_test)

model.score(X_train, y_train)

model.predict_proba(x_train[9:10])

accuracy_score(y_test, y_pred)

### Creating pickle ffile
pickle_out=open("model.pkl","wb")
pickle.dump(model, pickle_out)
pickle_out.close()






