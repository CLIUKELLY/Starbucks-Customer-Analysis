#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:55:55 2023

@author: kellyliu
"""

# =============================================================================
# Imports
# =============================================================================

import pandas as pd
# Load the datasets
portfolio = pd.read_csv('portfolio.csv')
profile = pd.read_csv('profile.csv')
transcript = pd.read_csv('transcript.csv')

# Show the basic information and the first few rows of each dataset
info_portfolio = portfolio.info()
head_portfolio = portfolio.head()

info_profile = profile.info()
head_profile = profile.head()

info_transcript = transcript.info()
head_transcript = transcript.head()

(info_portfolio, head_portfolio, info_profile, head_profile, info_transcript, head_transcript)

# =============================================================================
# Check & Drop missing value
# =============================================================================

missing_values = profile.isnull().sum()
missing_values = missing_values[missing_values > 0]

# Missing value in profile
# gender    2175
# income    2175
# no missing value in portfolio and transcript

# Test 1, drop alll the NAs
profile_cleaned = profile.dropna()

(missing_values, profile.shape, profile_cleaned.shape)
# Reduced the row from 17000 to 14825 rows

# =============================================================================
# EDA: Distribution + Summary Statistics
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Function to plot the distribution of numerical and categorical variables
def plot_distributions(df, numerical_columns, categorical_columns):
    fig, axes = plt.subplots(len(numerical_columns) + len(categorical_columns), 1, figsize=(10, 5 * (len(numerical_columns) + len(categorical_columns))))

    # Plot numerical distributions
    for i, col in enumerate(numerical_columns):
        sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')

    # Plot categorical distributions
    for i, col in enumerate(categorical_columns):
        sns.countplot(data=df, x=col, ax=axes[len(numerical_columns) + i])
        axes[len(numerical_columns) + i].set_title(f'Distribution of {col}')

    plt.tight_layout()
    plt.show()

# 1. Distributions
plot_distributions(portfolio, ['reward', 'difficulty', 'duration'], ['offer_type'])
plot_distributions(profile_cleaned, ['age', 'income'], ['gender'])
plot_distributions(transcript, ['time'], ['event'])

# 2. Summary Statistics
summary_portfolio = portfolio.describe(include='all')
summary_profile = profile_cleaned.describe(include='all')
summary_transcript = transcript.describe(include='all')

(summary_portfolio, summary_profile, summary_transcript)

# =============================================================================
# Customer segmentation predictor calculation
# =============================================================================

# Check the unique event types in the 'event' column
unique_events = transcript['event'].unique()
unique_events

# Predictor 1: offer completed number per customer
offer_completed_data = transcript[transcript['event'] == 'offer completed']
offer_completed_per_person = offer_completed_data.groupby('person').size()
offer_completed_per_person_df = offer_completed_per_person.reset_index(name='offer_completed_count')
offer_completed_per_person_df.shape

# Predictor 2: Transaction amount per customer
transaction_value_data = transcript[transcript['event'] == 'transaction']
import ast
transcript['amount'] = transcript['value'].apply(lambda x: ast.literal_eval(x).get('amount', 0))
transaction_value_per_person = transcript.groupby('person')['amount'].sum()
transaction_value_per_person_df = transaction_value_per_person.reset_index(name='transaction_value_amount')
transaction_value_per_person_df.shape

# Predictor 3: Offer received number per person
offer_received_data = transcript[transcript['event'] == 'offer received']
offer_received_per_person = offer_received_data.groupby('person').size()
offer_received_per_person_df = offer_received_per_person.reset_index(name='offer_received_count')
offer_received_per_person_df.shape

# Predictor 4: Offer reviewed number per person
offer_viewed_data = transcript[transcript['event'] == 'offer viewed']
offer_viewed_per_person = offer_viewed_data.groupby('person').size()
offer_viewed_per_person_df = offer_viewed_per_person.reset_index(name='offer_viewed_count')
offer_viewed_per_person_df.shape

# Merge the predictors into a single DataFrame on the 'person' column
predictors_df = offer_received_per_person_df.merge(offer_viewed_per_person_df, on='person', how='left').merge(offer_completed_per_person_df, on='person', how='left').merge(transaction_value_per_person_df, on='person', how='left')
predictors_df.fillna(0, inplace=True)

# Predictor 5: offer completion rate per person (completed/received)
predictors_df['offer_completion_rate'] = predictors_df['offer_completed_count'] / predictors_df['offer_received_count']
predictors_df['offer_completion_rate'].fillna(0, inplace=True)


# =============================================================================
# Run K-means to segment customer
# =============================================================================
X = predictors_df

numerical_features = predictors_df.drop(['person'], axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(numerical_features)
X_std_df = pd.DataFrame(X_std, columns=numerical_features.columns)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(X_std_df)

# Add the cluster labels to the original predictors_df DataFrame
predictors_df['cluster'] = cluster_labels

# Plot the clusters
sns.scatterplot(data=predictors_df, x='offer_completed_count', y='transaction_value_amount', hue='cluster')
plt.title('K-means Clustering with 4 clusters')
plt.xlabel('Offer Completed Count')
plt.ylabel('Transaction Value Amount')
plt.show()



































