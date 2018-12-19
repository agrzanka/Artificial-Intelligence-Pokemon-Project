import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np


def logistic_regression():
    df = pd.read_csv("Pokemontree.csv")

        #chart for data given at the begin
    # plt.scatter(df.Attack, df.Over, marker ='+', color='red')
    # plt.show()

    x_train, x_test, y_train, y_test = train_test_split(df[['Attack']], df.Over,test_size=0.1)
    print(x_test)
    model = LogisticRegression()
    model.fit(x_train, y_train)

    print(model.predict(x_test))
    print(model.score(x_test, y_test))
    print(model.predict_proba(x_test))

    single_test = [45]
    single_test = np.array(single_test).reshape(-1, 1)
    is_higher_than_1200_single = model.predict(single_test)
    print(is_higher_than_1200_single)

    wide_test = [i + 1 for i in range(200)]
    wide_test = np.array(wide_test).reshape(-1, 1)
    is_higher_than_1200_wide = model.predict(wide_test)

    plt.scatter(wide_test, is_higher_than_1200_wide, marker ='+', color='red')
    plt.show()

def main():
    logistic_regression()


if __name__ == "__main__":
    main()