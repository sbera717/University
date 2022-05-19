import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


def data_split(data,ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    test_set_size=int(len(data)*ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

if __name__ == "__main__":
    df=pd.read_csv('dataset.csv')
    df = df.dropna()
    train,test=data_split(df,0.5)
    X_train=train[['cough','fever','sore_throat','shortness_of_breath','head_ache','age_60_and_above','gender']].to_numpy()
    X_test=test[['cough','fever','sore_throat','shortness_of_breath','head_ache','age_60_and_above','gender']].to_numpy()
    
    Y_train=train[['infectionProb']].to_numpy().reshape()
    Y_test=test[['infectionProb']].to_numpy().reshape()
    
    clf=LogisticRegression()
    clf.fit(X_train,Y_train)


    file=open('covid.pkl','wb')
    pickle.dump(clf,file)
    file.close()
