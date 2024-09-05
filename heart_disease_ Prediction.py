import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#### loading the data into dataframe        ####
df = pd.read_csv('framingham.csv')
df = df.dropna()
# print(len(df.columns))

####        training data       ####
X = df.drop(columns=['TenYearCHD'])
####        predicting data         ####
Y = df.TenYearCHD
####        spliting the data into train and test part      ####
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state=42)

####        training the LogisticRegression model       ####
model = LogisticRegression(max_iter= 5000)

####        fiting the x and y train data into model        ####
model.fit(x_train,y_train)

####        predicting the disease rate         ####
pred = model.predict(x_train.sample(100))
print("\nthe predicted values for random 100 sample\n\n",pred)

count = 0
for i in pred:
    if i == 1:
        count += 1
rate = (count / 100)*100
print(f"\nOut of 100 randomly sampled individuals, {count} were predicted to have heart disease\n")
print(f"predict rate of percentage= {int(rate)}% heart disease")

####        printing the score of the model         ####
print("\n prediction score of the model\n",f"{int(model.score(x_test,y_test)*100)}%")
