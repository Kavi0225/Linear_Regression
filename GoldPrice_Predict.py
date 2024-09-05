import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Gold Price Prediction.csv')
####          droping the nan rows            ####
df = df.dropna()
####          traing data          ####
x = df[['Monthly Inflation Rate', 'EFFR Rate', 'Volume ','Treasury Par Yield Month', 'Treasury Par Yield Two Year','Treasury Par Yield Curve Rates (10 Yr)', 'DXY', 'SP Open', 'VIX','Crude']]
####          output data          ####
y = df.Price
####          creating the model using LinearRegression          ####
model = LinearRegression()
####          fit the training data form dataframe             ####
model.fit(x,y)
####          prediction using trained data             ####
op = model.predict(x)
print("\n####       predicted values        ####\n")
print(op)
####          storing the output data in the new columns in the dataframe           ####
df['output'] = op
####          adding the predicted output in the csv file             ####
# df.to_csv('Gold Price Prediction.csv',index = False)
####        calculating the deviation       ####
df['deviation'] = df['Price'] - df['output']
####        inserting the new columns to the csv file       ####
# df.to_csv('Gold Price Prediction.csv',index=False)
print("\n####        after the calculation of deviation          #### \n")
print(df.head())