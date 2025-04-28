import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

## train_test_split allows us to test accuracy of model on data already available

## randomforestregressor is the actual ML model
## builds many decision trees where each tree gives a prediction. the 'forest' of the trees avgs the trees' predictions to make a final prediction. 

def train_model(df):
    X = df.drop(columns = ['log_revenue'])
    Y = df['log_revenue']
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1) # splits 80% of data for training and 20% for testing. 
    
    model = RandomForestRegressor(n_estimators = 150, random_state=1)  # i read 100-300 trees is a good number to use for a dataset of 5000 movies so i chose 150 for n_predictors 
    model.fit(X_train, y_train)
    
    y_predicted = model.predict(X_test)

    return model, X_test, y_test, y_predicted