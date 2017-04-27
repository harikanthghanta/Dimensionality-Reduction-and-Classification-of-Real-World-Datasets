# Required Python Machine learning Packages
import pandas as pd
import numpy as np

# For preprocessing the data
from sklearn import preprocessing
from sklearn.preprocessing import Imputer

# To split the dataset into train and test datasets
from sklearn.cross_validation import train_test_split
#import naiveBayes as gnb


adult_df = pd.read_csv('adult_data.csv', header = None, delimiter=' *, *', engine='python')

adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                	'marital_status', 'occupation', 'relationship',
                	'race', 'sex', 'capital_gain', 'capital_loss',
                	'hours_per_week', 'native_country', 'income']

# for value in ['workclass', 'education',
#       	'marital_status', 'occupation',
#       	'relationship','race', 'sex',
#       	'native_country', 'income']:
# 	print value,":", sum(adult_df[value] == '?')

adult_df_rev = adult_df
	 
adult_df_rev.describe(include= 'all')
for value in ['workclass', 'education',
      	'marital_status', 'occupation',
      	'relationship','race', 'sex',
      	'native_country', 'income']:
	adult_df_rev[value].replace(['?'], [adult_df_rev.describe(include='all')[value][2]], inplace='True')

le = preprocessing.LabelEncoder()
workclass_cat = le.fit_transform(adult_df.workclass)
education_cat = le.fit_transform(adult_df.education)
marital_cat   = le.fit_transform(adult_df.marital_status)
occupation_cat = le.fit_transform(adult_df.occupation)
relationship_cat = le.fit_transform(adult_df.relationship)
race_cat = le.fit_transform(adult_df.race)
sex_cat = le.fit_transform(adult_df.sex)
native_country_cat = le.fit_transform(adult_df.native_country)
income_cat = le.fit_transform(adult_df.income)

adult_df_rev['workclass_cat'] = workclass_cat
adult_df_rev['education_cat'] = education_cat
adult_df_rev['marital_cat'] = marital_cat
adult_df_rev['occupation_cat'] = occupation_cat
adult_df_rev['relationship_cat'] = relationship_cat
adult_df_rev['race_cat'] = race_cat
adult_df_rev['sex_cat'] = sex_cat
adult_df_rev['native_country_cat'] = native_country_cat
adult_df_rev['income_cat'] = income_cat
#print adult_df_rev
#drop the old categorical columns from dataframe
dummy_fields = ['workclass', 'education', 'marital_status',
              	'occupation', 'relationship', 'race',
              	'sex', 'native_country', 'income']
adult_df_rev = adult_df_rev.drop(dummy_fields, axis = 1)

adult_df_rev = adult_df_rev.reindex_axis(['age', 'workclass_cat', 'fnlwgt', 'education_cat',
                                	'education_num', 'marital_cat', 'occupation_cat',
                                	'relationship_cat', 'race_cat', 'sex_cat', 'capital_gain',
                                	'capital_loss', 'hours_per_week', 'native_country_cat',
                                	'income_cat'], axis= 1)

num_features = ['age', 'workclass_cat', 'fnlwgt', 'education_cat', 'education_num',
            	'marital_cat', 'occupation_cat', 'relationship_cat', 'race_cat',
            	'sex_cat', 'capital_gain', 'capital_loss', 'hours_per_week',
            	'native_country_cat', 'income_cat']
print adult_df_rev
adult_df_rev.to_csv('./temp1.csv', sep=',', encoding='utf-8')
scaled_features = {}
for each in num_features:
	mean, std = adult_df_rev[each].mean(), adult_df_rev[each].std()
	scaled_features[each] = [mean, std]
	adult_df_rev.loc[:, each] = (adult_df_rev[each] - mean)/std

#print adult_df_rev
#adult_df_rev.to_csv('./temp.csv', sep=',', encoding='utf-8')
