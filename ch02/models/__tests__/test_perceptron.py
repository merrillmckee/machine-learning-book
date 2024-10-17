import numpy as np
import pytest

from ch02.models.perceptron import Perceptron


@pytest.fixture
def n_features():
    return 4


@pytest.fixture
def n_examples():
    return 7


@pytest.fixture
def perceptron(n_features):
    return Perceptron(n_features=n_features)


@pytest.fixture
def x_data(n_examples, n_features):
    rng = np.random.default_rng(seed=42)
    x_ = rng.random(size=(n_examples, n_features), dtype=float)
    return x_


@pytest.fixture
def y_data(n_examples):
    return np.array([float(i) % 2 for i in range(n_examples)])


def test_perceptron(perceptron):
    assert isinstance(perceptron, Perceptron)


def test_perceptron_fit(perceptron, x_data, y_data):
    # arrange
    original_weights = perceptron.w_.copy()

    # act
    perceptron.fit(x_data, y_data)

    # assert
    assert isinstance(perceptron, Perceptron)
    assert perceptron.w_.shape == original_weights.shape
    fit_labels = perceptron.predict(x_data)
    assert y_data == pytest.approx(fit_labels)


def test_perceptron_net_input(perceptron, x_data):
    y_hat = perceptron.net_input(x_data)
    assert len(y_hat) == x_data.shape[0]


def test_perceptron_predict(perceptron, x_data):
    y_label = perceptron.predict(x_data)
    assert len(y_label) == x_data.shape[0]


if __name__ == "__main__":
    pytest.main()
