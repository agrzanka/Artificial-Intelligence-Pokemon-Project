import pandas as panda
from sklearn import linear_model
from sklearn.metrics import r2_score

def linearPredictionPokemon():
    df = panda.read_csv("Pokemonedit.csv", sep=';')
    reg = linear_model.LinearRegression()
    reg.fit(df[['HP', 'Attack', 'Defense']], df.CP)

    y_pred = reg.predict(df[['HP', 'Attack', 'Defense']])

    print('Variance score: %.2f' % r2_score(df.CP, y_pred))
    print(reg.coef_)
    print(reg.intercept_)

def main():
    linearPredictionPokemon()


if __name__ == "__main__":
    main()
