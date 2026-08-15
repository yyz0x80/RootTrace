from benchmark.user_service import UserService


def test_register_and_find_user() -> None:
    service = UserService()
    created = service.register("person@example.com")

    assert service.find("person@example.com") == created

