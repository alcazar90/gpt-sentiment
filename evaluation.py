import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
)

def auc_score(test_set, predicted_set):
    high_predicted = np.array([prediction[2] for prediction in predicted_set])
    medium_predicted = np.array(
        [prediction[1] for prediction in predicted_set]
    )
    low_predicted = np.array([prediction[0] for prediction in predicted_set])
    inc_test = np.where(test_set == "incivilidad", 1.0, 0.0)
    odio_test = np.where(test_set == "odio", 1.0, 0.0)
    normal_test = np.where(test_set == "normal", 1.0, 0.0)
    auc_high = roc_auc_score(inc_test, high_predicted)
    auc_med = roc_auc_score(odio_test, medium_predicted)
    auc_low = roc_auc_score(normal_test, low_predicted)
    auc_w = (
        normal_test.sum() * auc_low
        + odio_test.sum() * auc_med
        + inc_test.sum() * auc_high
    ) / (normal_test.sum() + odio_test.sum() + inc_test.sum())
    return auc_w


def evaluate(predicted_probabilities, y_test, labels):
    """
        predicted_probabilities [np.array]: arreglo con las probabilidades de cada clase de los datos de test.
        y_test: [np.array]: arreglo con el verdadero ids de clases de los datos de test.
        labels: [np.array]: arreglo con los label de cada clase usados de encoding por el clasificador.
    """
    predicted_labels = [
        labels[np.argmax(item)] for item in predicted_probabilities
    ]

    print("Matriz de confusión")
    print(
        confusion_matrix(
            y_test, predicted_labels, labels=["normal", "odio", "incivilidad"]
        )
    )

    print("\nReporte de clasificación:\n")
    print(
        classification_report(
            y_test, predicted_labels, labels=["normal", "odio", "incivilidad"]
        )
    )
    # Reorder predicted probabilities array.
    labels = labels.tolist()

    predicted_probabilities = predicted_probabilities[
        :,
        [
            labels.index("normal"),
            labels.index("odio"),
            labels.index("incivilidad"),
        ],
    ]
    auc = round(auc_score(y_test, predicted_probabilities), 3)
    print("Métricas:\n\nAUC: ", auc, end="\t")
    kappa = round(cohen_kappa_score(y_test, predicted_labels), 3)
    print("Kappa:", kappa, end="\t")
    accuracy = round(accuracy_score(y_test, predicted_labels), 3)
    print("Accuracy:", accuracy)
    print("------------------------------------------------------\n")
    return np.array([auc, kappa, accuracy])
