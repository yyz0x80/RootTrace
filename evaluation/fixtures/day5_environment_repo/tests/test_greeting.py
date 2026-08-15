from environment_demo import greeting


def test_greeting() -> None:
    assert greeting("Ada") == "Hello, Ada!"

