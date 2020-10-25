import pytest
import numpy as np

Q = np.array([[0, 1, 2], [3, 4, 5]])
epsilon = 0.10

# choose action
def choose_action(row, p):
    if p < (1 - epsilon):
        i = np.argmax(Q[row, :])
    else:
        i = np.random.choice(Q.shape[0])
    return i

def test_choose_action():
	assert choose_action(1, 0.85) == 2
