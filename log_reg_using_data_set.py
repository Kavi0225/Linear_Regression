import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
digits = load_digits()
# print(digits)

print(digits.images[0],digits.images[1],digits.images[2])

# plt.gray()
# plt.matshow(digits.images[1])
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(digits.data,digits.target , test_size= .03)

model = LogisticRegression()
model.fit(x_train,y_train)

op = model.predict([digits.data[100]])
print(op)

scr = model.score(x_test,y_test)
print(scr)

plt.gray()
plt.matshow(digits.images[100])
plt.show()