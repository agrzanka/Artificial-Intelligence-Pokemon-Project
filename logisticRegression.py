import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

def logistic_regression():
    df = pd.read_csv("Pokemontree.csv")
    plt.scatter(df.Attack, df.Over,marker ='+', color='red')
    plt.show()
    x_train, x_test, y_train, y_test = train_test_split(df[['Attack']],df.Over,test_size=0.1)
    #print(x_test)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    yPrediction = model.predict(x_test)
    # print(yPrediction)
    score = model.score(x_test,y_test)
    # print(score)
    # print( model.predict_proba(x_test))
    temp = [150]
    temp = np.array(temp).reshape(-1,1)
    ile = model.predict(temp)
    print(ile)
    print('hahfa')


def main():
    logistic_regression()


if __name__ == "__main__":
    main()