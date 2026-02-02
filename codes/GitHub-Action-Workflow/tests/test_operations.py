from src.math_operations import add, sub


def test_add():
    assert add(2, 3) == 5
    assert add(-5, 5) == 0
    assert add(0, 7) == 7


def test_sub():
    assert sub(10, 2) == 8
    assert sub(6, 1) == 5
    assert sub(-5, 2) == -7
