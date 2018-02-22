import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = "cYe6mNAb8dtBp1N1mQc_"

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = ((df["Adj. High"]-df["Adj. Close"])/df["Adj. Close"])*100
df['PCT_CHANGE'] = ((df["Adj. Close"]-df["Adj. Open"])/df["Adj. Open"])*100

df = df[['Adj. Close','HL_PCT','PCT_CHANGE','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-999999, inplace=True)


#Predict data 1% of the length of the dataframe in advance.
forecast_out = int(math.ceil(0.01*len(df)))
#print("Days in advance: "+ str(forecast_out))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)


#Features column, drop the label in our data set.
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

X = preprocessing.scale(X) #Increases processing time.
y = np.array(df['label'])

#print("x is this long: "+ str(len(x)))
#print("Y is this long: " + str(len(y)))

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)
