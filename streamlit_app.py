import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile

# Page title
st.set_page_config(page_title='Tool Recommendation Engine Building', page_icon='🤖')
st.title('🤖 Tool Recommendation Engine Building')
st.info("In our journey to improve the Tubebuddy activation rate, we've realized how crucial smart tools like the Recommendation Engine are. At Tubebuddy, this isn’t just a feature; it’s our mission to connect creators with their ideal audience. This engine isn't just about improving user retention; it’s about creating a space where every recommendation feels like it's made just for you. We aim to use this engine to turn every interaction into an opportunity for creators to shine and audiences to find their next favorite content.")

with st.expander('About this app'):
  st.markdown('**Why Tool Recommendation Engine**')
  st.info("The data reveals that there isn't a single set of tools that guarantees high user retention. Relying on a singular recommendation approach risks losing users who could potentially have high long-term engagement. while there is a consistent set of tool combinations used weekly by some users, a significant portion of users change their tool usage weekly to maintain retention. This pattern indicates that basing recommendations solely on the top 10 or 20 tools each week could overlook many valuable opportunities. We need a recommendation engine that can dynamically adapt to these varied user behaviors and preferences to ensure we capture the full spectrum of engagement opportunities.")

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar and 1. Select a data set and 2. Adjust the model parameters by adjusting the various slider widgets. As a result, this would initiate the ML model building process, display the model results as well as allowing users to download the generated models and accompanying data.')


# Sidebar for accepting input parameters
with st.sidebar:
    st.header('1. Choose Use case')
    # Select example data
    st.markdown('**1.1 Use Case: Targeted Tool Recommendations for Less Active, Established Free Users**')
    example_data = st.toggle('Load example 1 data')
    if example_data:
        df = pd.read_csv('/workspaces/tool-recommendation-engine-building/model_data_high_retention.csv')
    st.markdown('**1.2 Use CaseTargeted Paid Tool Recommendations for High Active, Established Free Users**')
    example_data = st.toggle('Load example 2 data')
    if example_data:
        df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')


    st.header('2. Set Parameters')
    parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

    st.header('3. Recommendation Engine Setting')
    selected_tool = st.selectbox("Which tools do you want to predict?",
    ("Quick-Edit Toolbar", "Title Generator"),
    index=None,
    placeholder="Select tool")
    st.write('You selected:', selected_tool)
    
    user_predict = st.slider('How many users do you want to predict?', 0, 100, 10, 5)
    
# Initiate the model building process
if example_data: 
    with st.status("Running ...", expanded=True) as status:
    
        st.write("Loading data ...")
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]
            
        st.write("Splitting data ...")
        time.sleep(sleep_time)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-parameter_split_size)/100, random_state=parameter_random_state)
    
        st.write("Model training ...")
        time.sleep(sleep_time)

        if parameter_max_features == 'all':
            parameter_max_features = None
            parameter_max_features_metric = X.shape[1]
        
        rf = RandomForestRegressor(
                n_estimators=parameter_n_estimators,
                max_features=parameter_max_features,
                min_samples_split=parameter_min_samples_split,
                min_samples_leaf=parameter_min_samples_leaf,
                random_state=parameter_random_state,
                criterion=parameter_criterion,
                bootstrap=parameter_bootstrap,
                oob_score=parameter_oob_score)
        rf.fit(X_train, y_train)
        
        st.write("Applying model to make predictions ...")
        time.sleep(sleep_time)
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)
            
        st.write("Evaluating performance metrics ...")
        time.sleep(sleep_time)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        st.write("Displaying performance metrics ...")
        time.sleep(sleep_time)
        parameter_criterion_string = ' '.join([x.capitalize() for x in parameter_criterion.split('_')])
        #if 'Mse' in parameter_criterion_string:
        #    parameter_criterion_string = parameter_criterion_string.replace('Mse', 'MSE')
        rf_results = pd.DataFrame(['Random forest', train_mse, train_r2, test_mse, test_r2]).transpose()
        rf_results.columns = ['Method', f'Training {parameter_criterion_string}', 'Training R2', f'Test {parameter_criterion_string}', 'Test R2']
        # Convert objects to numerics
        for col in rf_results.columns:
            rf_results[col] = pd.to_numeric(rf_results[col], errors='ignore')
        # Round to 3 digits
        rf_results = rf_results.round(3)
        
    status.update(label="Status", state="complete", expanded=False)

    # Display data info
    st.header('Input data', divider='rainbow')
    col = st.columns(4)
    col[0].metric(label="No. of samples", value=X.shape[0], delta="")
    col[1].metric(label="No. of X variables", value=X.shape[1], delta="")
    col[2].metric(label="No. of Training samples", value=X_train.shape[0], delta="")
    col[3].metric(label="No. of Test samples", value=X_test.shape[0], delta="")
    
    with st.expander('Initial dataset', expanded=True):
        st.dataframe(df, height=210, use_container_width=True)
    with st.expander('Train split', expanded=False):
        train_col = st.columns((3,1))
        with train_col[0]:
            st.markdown('**X**')
            st.dataframe(X_train, height=210, hide_index=True, use_container_width=True)
        with train_col[1]:
            st.markdown('**y**')
            st.dataframe(y_train, height=210, hide_index=True, use_container_width=True)
    with st.expander('Test split', expanded=False):
        test_col = st.columns((3,1))
        with test_col[0]:
            st.markdown('**X**')
            st.dataframe(X_test, height=210, hide_index=True, use_container_width=True)
        with test_col[1]:
            st.markdown('**y**')
            st.dataframe(y_test, height=210, hide_index=True, use_container_width=True)

    # Zip dataset files
    df.to_csv('dataset.csv', index=False)
    X_train.to_csv('X_train.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    list_files = ['dataset.csv', 'X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv']
    with zipfile.ZipFile('dataset.zip', 'w') as zipF:
        for file in list_files:
            zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

    with open('dataset.zip', 'rb') as datazip:
        btn = st.download_button(
                label='Download ZIP',
                data=datazip,
                file_name="dataset.zip",
                mime="application/octet-stream"
                )
    
    # Display model parameters
    st.header('Model parameters', divider='rainbow')
    parameters_col = st.columns(3)
    parameters_col[0].metric(label="Data split ratio (% for Training Set)", value=parameter_split_size, delta="")
    parameters_col[1].metric(label="Number of estimators (n_estimators)", value=parameter_n_estimators, delta="")
    parameters_col[2].metric(label="Max features (max_features)", value=parameter_max_features_metric, delta="")
    
    # Display feature importance plot
    importances = rf.feature_importances_
    feature_names = list(X.columns)
    forest_importances = pd.Series(importances, index=feature_names)
    df_importance = forest_importances.reset_index().rename(columns={'index': 'feature', 0: 'value'})
    
    bars = alt.Chart(df_importance).mark_bar(size=40).encode(
             x='value:Q',
             y=alt.Y('feature:N', sort='-x')
           ).properties(height=250)

    performance_col = st.columns((2, 0.2, 3))
    with performance_col[0]:
        st.header('Model performance', divider='rainbow')
        st.dataframe(rf_results.T.reset_index().rename(columns={'index': 'Parameter', 0: 'Value'}))
    with performance_col[2]:
        st.header('Feature importance', divider='rainbow')
        st.altair_chart(bars, theme='streamlit', use_container_width=True)

    # Prediction results
    st.header('Prediction results', divider='rainbow')
    s_y_train = pd.Series(y_train, name='actual').reset_index(drop=True)
    s_y_train_pred = pd.Series(y_train_pred, name='predicted').reset_index(drop=True)
    df_train = pd.DataFrame(data=[s_y_train, s_y_train_pred], index=None).T
    df_train['class'] = 'train'
        
    s_y_test = pd.Series(y_test, name='actual').reset_index(drop=True)
    s_y_test_pred = pd.Series(y_test_pred, name='predicted').reset_index(drop=True)
    df_test = pd.DataFrame(data=[s_y_test, s_y_test_pred], index=None).T
    df_test['class'] = 'test'
    
    df_prediction = pd.concat([df_train, df_test], axis=0)
    
    prediction_col = st.columns((2, 0.2, 3))
    
    # Display dataframe
    with prediction_col[0]:
        st.dataframe(df_prediction, height=320, use_container_width=True)

    # Display scatter plot of actual vs predicted values
    with prediction_col[2]:
        scatter = alt.Chart(df_prediction).mark_circle(size=60).encode(
                        x='actual',
                        y='predicted',
                        color='class'
                  )
        st.altair_chart(scatter, theme='streamlit', use_container_width=True)

    
# Ask for CSV upload if none is detected
else:
    st.info('Load example use case to get started!')
