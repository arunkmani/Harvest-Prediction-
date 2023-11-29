from random import random
import pandas as pd 
import numpy as np 
import random
import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
def converter(row):
    if row['Period'] == 'Monthly':
        year=row['Year']
        month=row['PeriodNo']
        return datetime.date(year,month,1).isocalendar()[1]
    else:
        return row['PeriodNo']
list1=list(range(2014,2024))
print(list1)
df1 = pd.DataFrame()
for item in list1:
    name=str(item)+'.csv'
    df=pd.read_csv(name)
    df1= pd.concat((df1,df),axis=0)
df1=df1.replace('<31','31')
df2=pd.to_datetime(df1['Date/Time'], errors ='coerce')
df2=df2.to_frame()
#df2.drop(df.index[3598:3651],inplace=True)
df1['Week']= df2['Date/Time'].dt.week
df1['Year']= df2['Date/Time'].dt.year
df1=df1.drop(columns=['Longitude (x)','Latitude (y)','Station Name','Climate ID','Data Quality'],axis=1)
df1=df1.drop(columns=['Max Temp Flag','Mean Temp Flag','Heat Deg Days Flag','Cool Deg Days Flag'],axis=1)
df1=df1.drop(columns=['Total Rain (mm)','Total Rain Flag','Total Snow (cm)','Total Snow Flag'],axis=1)
df1=df1.drop(columns=['Min Temp Flag','Total Precip Flag','Snow on Grnd Flag','Dir of Max Gust Flag','Spd of Max Gust Flag'],axis=1)
df1=df1.drop(columns=['Month','Day'],axis=1)
df_final=pd.DataFrame()
df_filtered=pd.DataFrame(columns= df1.columns)
for item in list1:
    for week_num in list(range(1,54)):
        if item == 2023 and week_num > 45:
            ryery=0
        else:
            df_final=df1.loc[(df1['Year'] == item) & (df1['Week'] == week_num)]
            if df_final.shape[0] > 0:
                temp= df_final.iloc[[random.randrange(df_final.shape[0])]]
                df_filtered = pd.concat((df_filtered,temp),axis=0)
df_filtered.fillna(0)
df_historical= pd.read_csv('historical.csv')
df_historical = df_historical.dropna(subset=['Price'])
df_historical['PeriodNo']=df_historical.apply(converter,axis=1)
df_historical=df_historical.drop(columns=['Period'],axis=1)
df_historical.rename(columns={"PeriodNo":"Week"},inplace=True)
df_merged = pd.merge(df_historical, df_filtered, on=['Year','Week'], how='inner')
df_canola= df_merged.loc[df_merged['Crop'] == 'Wheat, Red Winter']
df_canola=df_canola.dropna()
y=df_canola['Price']
df_canola = df_canola.drop(columns=['Price','Crop','Date/Time'],axis=1)
X_train,X_test,y_train,y_test= train_test_split(df_canola,y,test_size=.2,random_state=42)
degree = 1 
poly_features = PolynomialFeatures(degree=degree)
poly_regression_model = make_pipeline(poly_features, LinearRegression())
poly_regression_model.fit(X_train, y_train)
y_pred = poly_regression_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
final=pd.DataFrame(columns=['test','predictions'])
final['test']=y_test
final['predictions'] = y_pred
final.to_csv('out.csv')
y_test=y_test.to_numpy()
plt.plot(y_test,label='Actual Data')
plt.plot(y_pred,label='Predcited Values')
plt.ylabel('Price in $')
plt.xlabel('Number of predictions')
plt.legend(loc="upper left")
plt.show()
