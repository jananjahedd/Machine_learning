import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine


# this part of the code is for all features and the generation of the heatmap
wine = datasets.load_wine() 
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target

X_train, X_test, y_train, y_test = train_test_split(wine_df.drop('target',axis=1), wine_df['target'], test_size=0.3, random_state=42)

logistic_model = LogisticRegression()
logistic_model.fit(X_train,y_train)
predictions = logistic_model.predict(X_test)

print(classification_report(y_test, predictions))

correlation_matrix = wine_df[selected_features].corr()
print(correlation_matrix)
