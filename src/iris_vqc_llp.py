import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms.classifiers import VQC


def main() -> None:
    data = load_iris()
    X = data.data
    y = data.target
    train_features, test_features, train_labels, test_labels = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    num_features = train_features.shape[1]

    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
    ansatz = RealAmplitudes(num_qubits=num_features, reps=3)

    optimizer = COBYLA(maxiter=100)
    sampler = Sampler()

    objective_func_vals = []

    def callback_graph(_weights, obj_func_eval):
        objective_func_vals.append(obj_func_eval)

    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback_graph,
    )

    start = time.time()
    vqc.fit(train_features, train_labels)
    elapsed = time.time() - start

    accuracy = vqc.score(test_features, test_labels)
    print(f"Training time: {round(elapsed)} seconds")
    print(f"Test accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
