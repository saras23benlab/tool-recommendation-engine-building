import streamlit as st
import pandas as pd
from surprise import accuracy
# Class to parse a file containing ratings, data should be in structure - user; item; rating
from surprise.reader import Reader
# Class for loading datasets
from surprise.dataset import Dataset
# For splitting the rating data in train and test datasets
from surprise.model_selection import train_test_split
# For implementing similarity-based recommendation system
from surprise.prediction_algorithms.knns import KNNBasic
from collections import defaultdict
import time
import random

# Page title
st.set_page_config(page_title='Tool Recommendation Engine Building', page_icon='🤖')
st.title('🤖 Tool Recommendation Engine Building')
st.info("In our journey to improve the Tubebuddy activation rate, we've realized how crucial smart tools like the Recommendation Engine are. At Tubebuddy, this isn’t just a feature; it’s our mission to connect creators with their ideal audience. This engine isn't just about improving user retention; it’s about creating a space where every recommendation feels like it's made just for you. We aim to use this engine to turn every interaction into an opportunity for creators to shine and audiences to find their next favorite content.")

with st.expander('About this app'):
  st.markdown('**Why Tool Recommendation Engine**')
  st.info("The data reveals that there isn't a single set of tools that guarantees high user retention. Relying on a singular recommendation approach risks losing users who could potentially have high long-term engagement. while there is a consistent set of tool combinations used weekly by some users, a significant portion of users change their tool usage weekly to maintain retention. This pattern indicates that basing recommendations solely on the top 10 or 20 tools each week could overlook many valuable opportunities. We need a recommendation engine that can dynamically adapt to these varied user behaviors and preferences to ensure we capture the full spectrum of engagement opportunities.")
  st.info("Domo Link: https://bengroup.domo.com/page/196225318 ")
  st.markdown('**Methodology**')
  st.image('https://raw.githubusercontent.com/saras23benlab/tool-recommendation-engine-building/master/user-user.png', caption='User-User Collaborative filtering')
  st.markdown('**How does it work**')
  st.info("Lucid chart: https://lucid.app/lucidchart/e19682f5-5e27-4af9-8cb6-10a82b86b2c3/edit?viewport_loc=-3704%2C-956%2C4736%2C2564%2C.Q4MUjXso07N&invitationId=inv_0e18c086-b21a-40f4-9732-892507a87971")
  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar, Choose Use Case, Set Parameters, and Recommendation Engine Setting. As a result, this would show all the predict tools for selected users')


# Sidebar for accepting input parameters
with st.sidebar:
    st.header('1. Choose Use case')

    # Select example data
    st.markdown('**1.1 Use Case: Targeted Tool Recommendations for Less Active, Established Free Users**')
    example_data_1 = st.checkbox('Load example 1 data')

    st.markdown('**1.2 Use Case: Targeted Paid Tool Recommendations for High Active, Established Free Users**')
    example_data_2 = st.checkbox('Load example 2 data')

    st.header('2. Set Parameters')
    parameter_split_size = st.slider('Data split ratio (% for Testing Set)', 10, 90, 80, 5)

    sleep_time = st.slider('Sleep time', 0, 3, 0) 
# Initiate the model building process
 # Load data based on checkbox selection
df = None
df_prediction = None

if example_data_1:
    df = pd.read_csv('https://raw.githubusercontent.com/saras23benlab/tool-recommendation-engine-building/master/model_data.csv')
    df = df.drop(['Unnamed: 0'], axis=1)
    df = df[(df['license'] == 'free') & (df['recur_type'] == 'free') & (df['MKT_persona_category'] == 'Established Creator')]
    df = df[['user_id', 'tool', 'tool_usage']]
    df_prediction = df

if example_data_2:
    df = pd.read_csv('https://raw.githubusercontent.com/saras23benlab/tool-recommendation-engine-building/master/model_data.csv')
    df = df.drop(['Unnamed: 0'], axis=1)
    df_sample1 = df[(df['license'] == 'free') & (df['recur_type'] == 'free') & (df['MKT_persona_category'] == 'Established Creator')]
    df_sample1 = df_sample1[['user_id', 'tool', 'tool_usage']]
    hr = pd.read_csv('https://raw.githubusercontent.com/saras23benlab/tool-recommendation-engine-building/master/model_data_high_retention.csv')
    hr = hr[['user_id', 'license', 'recur_type']]
    hr = hr.drop_duplicates()
    hr_free = hr[hr['license'] == 'free']
    high_retention_free_user_list = hr_free['user_id'].tolist()
    df_sample2_paid = df[(df['license'].isin(['legend','star','pro'])) & (df['recur_type'].isin(['all_time_paid', 'free -> paid'])) & (df['MKT_persona_category'] == 'Established Creator')]
    df_sample2_paid = df_sample2_paid[['user_id', 'tool', 'tool_usage']]
    #paid = list(set(df_sample2_paid['user_id'].tolist()))
    #random.seed(42)
    #selected_elements = random.sample(paid, 300)
    #df_sample2_paid = df_sample2_paid[df_sample2_paid['user_id'].isin(selected_elements)]
    temp = df_sample1[['user_id', 'tool', 'tool_usage']]
    df_sample2_hr_free = temp[temp['user_id'].isin(high_retention_free_user_list)]
    df = pd.concat([df_sample2_paid, df_sample2_hr_free], axis=0)
    df_prediction = df_sample2_hr_free
 
if df is not None and df_prediction is not None:
    with st.status("Running ...", expanded=True) as status:
    
        st.write("Loading data ...")
        time.sleep(sleep_time)
        # Instantiating Reader scale with expected rating scale
        reader = Reader(rating_scale = (1, 5))

       # Initialize an empty column for rate
        df['rate'] = pd.NA

        # Define quantiles
        quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1]

        # Define bin labels
        bin_labels = [1, 2, 3, 4, 5]

        # Function to apply quantile binning to each group
        def quantile_binning(group):
            # Calculate bin edges for the current group, dropping duplicates
            bin_edges = group['tool_usage'].quantile(quantiles).round(0).unique().tolist()
            bin_edges.sort()  # Ensure the bin edges are sorted
            print(group['tool'].unique().tolist(), bin_edges)

            # Adjust bin labels based on the number of unique bin edges - 1
            unique_bin_labels = range(1, len(bin_edges))

            # Bin the group's tool_usage column and assign it to the rate column
            group['rate'] = pd.cut(group['tool_usage'], bins=bin_edges, labels=unique_bin_labels, include_lowest=True)
            return group

        # Apply the quantile binning function to each group
        df = df.groupby('tool').apply(quantile_binning)

        # Loading the rating dataset
        data = Dataset.load_from_df(df[['user_id', 'tool', 'rate']], reader)
        # Splitting the data into train and test datasets
        trainset, testset = train_test_split(data, test_size = 0.01 * parameter_split_size, random_state = 42)
        st.write("Model training ...")
        time.sleep(sleep_time)

        def precision_recall_at_k(model, k = 10, threshold = 1.5):
            """Return precision and recall at k metrics for each user"""

            # First map the predictions to each user
            user_est_true = defaultdict(list)

            # Making predictions on the test data
            predictions = model.test(testset)

            for uid, _, true_r, est, _ in predictions:
                user_est_true[uid].append((est, true_r))

            precisions = dict()
            recalls = dict()
            for uid, user_ratings in user_est_true.items():

                # Sort user ratings by estimated value
                user_ratings.sort(key = lambda x: x[0], reverse = True)

                # Number of relevant items
                n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

                # Number of recommended items in top k
                n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

                # Number of relevant and recommended items in top k
                n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                    for (est, true_r) in user_ratings[:k])

                # Precision@K: Proportion of recommended items that are relevant
                # When n_rec_k is 0, Precision is undefined. Therefore, we are setting Precision to 0 when n_rec_k is 0

                precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

                # Recall@K: Proportion of relevant items that are recommended
                # When n_rel is 0, Recall is undefined. Therefore, we are setting Recall to 0 when n_rel is 0

                recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

            rmse = accuracy.rmse(predictions)

            # Mean of all the predicted precisions are calculated
            precision = round((sum(prec for prec in precisions.values()) / len(precisions)), 3)

            # Mean of all the predicted recalls are calculated
            recall = round((sum(rec for rec in recalls.values()) / len(recalls)), 3)

            # Compute F1 score
            if precision + recall != 0:
                f1_score = round((2 * precision * recall) / (precision + recall), 3)
            else:
                f1_score = 0

            # Return the precision, recall, and F1 score
            return rmse, precision, recall, f1_score

       # Using the optimal similarity measure for user-user collaborative filtering
        sim_options = {'name': 'msd',
                    'user_based': True}

        # Creating an instance of KNNBasic with optimal hyperparameter values
        sim_user_user_optimized = KNNBasic(sim_options = sim_options, k = 50, min_k = 6, random_state = 1, verbose = False)

        # Training the algorithm on the train set
        sim_user_user_optimized.fit(trainset)

        st.write("Evaluating performance metrics ...")
        time.sleep(sleep_time)
        # Let us compute precision@k, recall@k, and F_1 score with k = 10
        precision_recall_at_k(sim_user_user_optimized)
        st.write("Tool Recommendation prediction ...")
        time.sleep(sleep_time)
        def n_users_not_interacted_with(n, data, tool):
            users_interacted_with_product = set(data[data['tool'] == tool]['user_id'])
            all_users = set(data['user_id'])
            return list(all_users.difference(users_interacted_with_product))[:n] # where n is the number of elements to get in the list

        
        def get_recommendations_for_users(data, user_ids, top_n, algo):
            all_recommendations = {}

            for user_id in user_ids:
                recommendations = []
                user_item_interactions_matrix = data.pivot(index='user_id', columns='tool', values='rate')
                non_interacted_tool = user_item_interactions_matrix.loc[user_id][user_item_interactions_matrix.loc[user_id].isnull()].index.tolist()

                for item_id in non_interacted_tool:
                    est = algo.predict(user_id, item_id).est
                    recommendations.append((item_id, est))

                recommendations.sort(key=lambda x: x[1], reverse=True)
                all_recommendations[user_id] = recommendations[:top_n]

            # Flatten the dictionary to a list suitable for DataFrame conversion
            recommendation_list = []
            for user_id, recommendations in all_recommendations.items():
                for tool_id, rating in recommendations:
                    recommendation_list.append({'user_id': user_id, 'tool_id': tool_id, 'rating': rating})

            # Convert the list to a DataFrame
            recommendations_df = pd.DataFrame(recommendation_list)

            return recommendations_df
        
        # Making top 5 recommendations for userId 4 using the similarity-based recommendation system
        unique_user_ids = df['user_id'].unique().tolist()
        recommendations = get_recommendations_for_users(df, unique_user_ids , 3, sim_user_user_optimized)
     
    status.update(label="Status", state="complete", expanded=False)
   

    # Model Metric
    # Let us compute precision@k, recall@k, and F_1 score with k = 10
    rmse, precision, recall, f1_score = precision_recall_at_k(sim_user_user_optimized)
    st.header('Model Performance')
    st.write(f'RMSE: {rmse}',)
    st.write(f'Precision: all the relevant tools {precision* 100:.2f}% are recommended.')  # Display the overall precision
    st.write(f'Recall: out of all the recommended tools {recall* 100:.2f}% are relevant.')  # Display the overall recall
    
     
    # Prediction results
    st.header('Prediction results', divider='rainbow')

# Use a container to span the full width
    with st.container():
        st.dataframe(recommendations, height=320)
    

    
# Ask for CSV upload if none is detected
else:
    st.info('Load example use case to get started!')
