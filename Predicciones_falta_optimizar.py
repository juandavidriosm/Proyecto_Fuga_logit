import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')


df_fuga_Y_final = pd.read_csv("Out/df_fuga_Y_final.csv")
df_fuga_completo = pd.read_csv("Out/df_fuga_completo.csv")

df_fuga_completo = pd.read_csv("Out/df_fuga_train.csv")
to_factor = list(df_fuga_completo.loc[:,df_fuga_completo.nunique() <= 4]);  
df_fuga_completo[to_factor] = df_fuga_completo[to_factor].astype('category')

Y =df_fuga_completo.Fuga
df_fuga_completo.drop("Fuga",axis = 1,inplace = True)

df_fuga_Y_final = df_fuga_Y_final[df_fuga_completo.columns]


pca = PCA(n_components=0.6)
logreg = LogisticRegression(C=0.001,penalty="l1", solver="liblinear")

pipe_model_interac = Pipeline(steps=[("pca", pca), ("logreg", logreg)])

pipe_model_interac.fit(df_fuga_completo,Y)

predictions_finales = pipe_model_interac.predict(df_fuga_Y_final)


pd.Series(predictions_finales).to_csv("Out/PrediccionesFinales")
