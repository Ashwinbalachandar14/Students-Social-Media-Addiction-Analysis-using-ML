import pandas as pd
import numpy as np
import statsmodels.api as sm

data=pd.read_csv("Students Social Media Addiction.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [2,3,4,6,7,10])],remainder='passthrough')
x=ct.fit_transform(x).toarray()

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
x=imputer.fit_transform(x)
imputer_y=SimpleImputer(missing_values=np.nan,strategy='mean')
y=imputer_y.fit_transform(y.reshape(-1,1))

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,train_size=0.8,random_state=1)



def backward_elimination(x,y,sl=0.05):
    nums=data.shape[1]
    for i in range(nums):
        regressor_ols = sm.OLS(y, x).fit()
        max_p_value=max(regressor_ols.pvalues).astype(float)
        if max_p_value > sl:
            max_p_delete=regressor_ols.pvalues.argmax()
            x=np.delete(x,max_p_delete,1)
        else:
            break
    return x
x=backward_elimination(x,y)


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10)
regressor.fit(X_train,Y_train)
Y_pre=regressor.predict(X_test)

import seaborn as sns
import matplotlib.pyplot as plt

platform_scores = data.groupby("Most_Used_Platform")["Addicted_Score"].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=platform_scores.index, y=platform_scores.values, palette="viridis")

plt.xlabel("Most Used Platform")
plt.ylabel("Average Addicted Score")
plt.title("Addiction Score by Most Used Social Media Platform")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()


