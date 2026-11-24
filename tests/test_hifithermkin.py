"""hifithermkin tests."""

import hifithermkin


def test_stub() -> None:
    """Stub test to ensure the test suite runs."""
    print(hifithermkin.__version__)  # noqa: T201


def test__greet() -> None:
    """Test the greet function."""
    assert hifithermkin.greet("World") == "Hello, World!"


def test__greet_jim() -> None:
    """Test the greet_jim function."""
    assert hifithermkin.greet_jim() == "Hello, Jim!"
