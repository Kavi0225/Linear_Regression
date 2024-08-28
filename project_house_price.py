import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import streamlit as st
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('C:/Users/demon/PycharmProjects/pythonProject/house_prices.csv')

# Data preprocessing
data = pd.get_dummies(data, drop_first=True)  # One-hot encode categorical variables

# Feature and target selection
X = data[['sqft_living', 'bedrooms', 'bathrooms']]  # Features
y = data['price']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model development
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open('../../../../PycharmProjects/pythonProject/house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the model for prediction
with open('../../../../PycharmProjects/pythonProject/house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')  # MSE is important for measuring prediction accuracy, penalizing larger errors
print(f'R-squared: {r2:.2f}')

# Streamlit app
st.title('House Price Predictor')

# Input fields for user to enter data
size = st.number_input('Size (in sq ft)')
bedrooms = st.number_input('Number of Bedrooms')
bathrooms = st.number_input('Number of Bathrooms')

# Prepare the input data for prediction
if st.button('Predict Price'):
    input_data = pd.DataFrame({
        'sqft_living': [size],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms]
    })
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # Predict the house price
    predicted_price = model.predict(input_data)[0]
    st.write(f'Predicted House Price: ${predicted_price:,.2f}')

    # Display R-squared score
    st.write(f'R-squared Score: {r2:.2f}')  # Measure of how well the model fits the data

    # Plotting results in Streamlit
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Actual vs Predicted Prices
    ax.scatter(y_test, y_pred, alpha=0.5, label='Actual vs Predicted Prices')

    # Highlight the predicted price
    ax.scatter(predicted_price, predicted_price, color='red', s=100, label='Predicted Price')

    ax.set_xlabel('Actual Prices')
    ax.set_ylabel('Predicted Prices')
    ax.set_title('Actual vs Predicted House Prices')
    ax.legend()

    st.pyplot(fig)