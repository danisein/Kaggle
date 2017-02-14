# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
def preprocessing(train_df, test_df):
	y_train = train_df.pop('SalePrice')
	y_train = np.log1p(y_train)
	alldf = pd.concat((train_df, test_df), axis = 0)
	#process category data
	alldf['MSSubClass'] = alldf['MSSubClass'].astype(str)
	all_dummy_df = pd.get_dummies(alldf)
	#process numeric data
	mean_cols = all_dummy_df.mean()
	test = all_dummy_df.isnull().sum()
	test.sort(ascending = False)
	print test.head()
	all_dummy_df = all_dummy_df.fillna(mean_cols)
	test = all_dummy_df.isnull().sum()
	test.sort(ascending = False)
	print test.head()

	num_cols = alldf.columns[alldf.dtypes != 'object']
	num_mean = all_dummy_df.loc[:, num_cols].mean()
	num_std = all_dummy_df.loc[:, num_cols].std()
	all_dummy_df.loc[:,num_cols] = (all_dummy_df.loc[:,num_cols] - num_mean) / num_std

	#split
	dummy_train_df = all_dummy_df.loc[train_df.index]
	dummy_test_df = all_dummy_df.loc[test_df.index]
	
	return dummy_train_df, dummy_test_df, y_train


def train(x_train, y_train):
	alphas = np.logspace(-3, 2, 50)
	test_scores = []
	print x_train.shape
	print y_train.shape
	for alpha in alphas:
		model = Ridge(alpha)
		score = cross_val_score(model, x_train, y_train, cv=10, scoring='mean_squared_error')
		print score
		test_scores.append(np.mean(score))
	print test_scores
	print np.argmax(test_scores)
	plt.plot(alphas, test_scores)
	plt.show()

	


if __name__ == '__main__':
	train_df = pd.read_csv("input/train.csv", index_col = 0)
	test_df = pd.read_csv("input/test.csv", index_col = 0)

	dummy_train_df, dummy_test_df, y_train = preprocessing(train_df, test_df)
	train(dummy_train_df.values, y_train.values)

