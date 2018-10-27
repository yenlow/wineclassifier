import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils

pd.set_option('display.max_colwidth',280)
pd.set_option('display.max_columns', 10)

#Load data
data = pd.read_csv('data/winemag-data_first150k.csv')
data.rename(columns={'Unnamed: 0':'id'}, inplace=True)
data.dtypes

data.variety = utils.rare2other(data.variety, min_sparsity=0.01, newvalue='other', missing='other', verbose=True, plot=True, plotn=20)
data.country = utils.rare2other(data.country, min_sparsity=0.01, newvalue='other', missing='other', verbose=True, plot=True, plotn=20)

data.price.hist(bins=50)
plt.show()


# Do some preprocessing to limit the # of wine varities in the dataset
data = data[pd.notnull(data['variety'])]
data = data[pd.notnull(data['price'])]
data = data[pd.notnull(data['points'])]
print("Dim: ", str(data.shape))   #(150930, 11)

#Reframe into a recommendation problem: if points > 90, accept recommendation
data['accept'] = np.where( data.points>=90, 1, 0)
data.accept.value_counts()
