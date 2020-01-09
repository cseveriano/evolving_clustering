from sklearn import preprocessing
from evolving import EvolvingClustering
from evolving.util import load_dataset


def main():
    X, y = load_dataset.load_dataset("gaussian")
#    X, y = load_dataset.load_dataset("s2")

    X = X[:1000, :8]
    y = y[:1000]

    standardized_X = preprocessing.scale(X)
    minmaxscaler = preprocessing.MinMaxScaler()
    minmaxscaler.fit(standardized_X)
    X = minmaxscaler.transform(standardized_X)

    evol_model = EvolvingClustering.EvolvingClustering(variance_limit=0.01, debug=True)
#    evol_model = EvolvingClustering2.EvolvingClustering2(rad=0.04, debug=True)
    evol_model.fit(X[:100])
    evol_model.fit(X[100:200])
    y_pred = evol_model.predict(X[:3000])

if __name__ == '__main__':
    main()