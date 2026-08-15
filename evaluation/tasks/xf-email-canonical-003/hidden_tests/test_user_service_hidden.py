import pytest
from benchmark.user_service import UserService
from benchmark.users import User


def test_user_factory_and_service_store_canonical_email() -> None:
    assert User.from_email("  Ada@Example.COM ").email == "ada@example.com"
    service = UserService()
    created = service.register("  Ada@Example.COM ")
    assert created.email == "ada@example.com"
    assert service.find("ADA@example.com") == created


def test_duplicate_canonical_email_is_rejected() -> None:
    service = UserService()
    service.register("ada@example.com")
    with pytest.raises(ValueError):
        service.register(" ADA@EXAMPLE.COM ")
