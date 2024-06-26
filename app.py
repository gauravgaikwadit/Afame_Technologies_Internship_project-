import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('Sales.csv')
    return data

# Data preprocessing
def preprocess_data(data):
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    
    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

# Train different models and evaluate them
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'ElasticNet Regression': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Support Vector Regression': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append((name, model, mse, r2))
    
    # Find the best model based on R^2 score
    best_model = max(results, key=lambda item: item[3])
    
    return best_model, results

# Streamlit app layout
def main():
    st.title('Sales Prediction Dashboard with Multiple Models')
    st.write("""
    ## Overview
    This dashboard predicts sales based on advertising expenditures in TV, Radio, and Newspaper, and compares different models.
    """)

    # Load data
    data = load_data()

    # Handle missing values
    if data.isnull().sum().sum() > 0:
        st.write("Data contains missing values. Filling missing values with the mean of respective columns.")
        data = data.fillna(data.mean())

    # Display the data
    st.write('### Sales Data')
    st.write(data.head())

    # Display data summary
    st.write('### Data Summary')
    st.write(data.describe())

    # Detect and remove outliers
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
    st.write('### Outlier Detection')
    st.write(outliers.sum())
    data = data[~outliers.any(axis=1)]

    # Data visualization
    st.write('### Data Visualization')
    fig_tv = px.scatter(data, x='TV', y='Sales', title='TV vs Sales')
    st.plotly_chart(fig_tv)
    fig_radio = px.scatter(data, x='Radio', y='Sales', title='Radio vs Sales')
    st.plotly_chart(fig_radio)
    fig_newspaper = px.scatter(data, x='Newspaper', y='Sales', title='Newspaper vs Sales')
    st.plotly_chart(fig_newspaper)

    # Correlation analysis
    st.write('### Correlation Matrix')
    corr_matrix = data.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect='auto', title='Correlation Matrix')
    st.plotly_chart(fig_corr)

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)

    # Train and evaluate models
    best_model, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Display model performance
    st.write('### Model Performance')
    results_df = pd.DataFrame(results, columns=['Model', 'Model Instance', 'Mean Squared Error', 'R^2 Score'])
    st.write(results_df.drop(columns=['Model Instance']))

    # Display best model
    st.write('### Best Model')
    st.write(f'The best model is: {best_model[0]} with R^2 Score: {best_model[3]:.4f}')

    # User input for prediction
    st.write('### Make a Prediction')
    tv = st.number_input('TV Advertising Expenditure ($)', min_value=0.0, max_value=300.0, value=100.0)
    radio = st.number_input('Radio Advertising Expenditure ($)', min_value=0.0, max_value=50.0, value=20.0)
    newspaper = st.number_input('Newspaper Advertising Expenditure ($)', min_value=0.0, max_value=100.0, value=30.0)

    # Make prediction
    if st.button('Predict'):
        new_data = pd.DataFrame({'TV': [tv], 'Radio': [radio], 'Newspaper': [newspaper]})
        new_data_scaled = scaler.transform(new_data)
        prediction = best_model[1].predict(new_data_scaled)
        st.write(f'### Predicted Sales: {prediction[0]:.2f}')
        
     # Optimization suggestions
    st.write('### Optimization Suggestions')
    st.write("""
    1. **Identify High-Impact Advertising Channels**: Allocate more budget to channels with the highest positive impact on sales.
    2. **Optimize Advertising Budget**: Use the prediction model to simulate different scenarios and find the optimal combination of expenditures.
    3. **Targeted Marketing Campaigns**: Develop targeted marketing campaigns to reach the most responsive customer segments.
    4. **Regular Performance Monitoring**: Continuously monitor and adjust strategies based on real-time data and predictions.
    5. **A/B Testing**: Conduct A/B testing for different advertising strategies and implement the most effective ones.
    6. **Seasonal Adjustments**: Analyze seasonal trends and adjust advertising expenditures during peak sales periods.
    7. **Cross-Promotions and Partnerships**: Partner with complementary brands for cross-promotions to increase reach and sales.
    8. **Content Quality and Messaging**: Ensure high-quality content and consistent messaging in advertisements to drive engagement and conversions.
    """)

if __name__ == '__main__':
    main()
