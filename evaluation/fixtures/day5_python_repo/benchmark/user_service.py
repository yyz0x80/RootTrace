"""In-memory user registration service."""

from benchmark.users import User, normalize_email


class UserService:
    def __init__(self) -> None:
        self._users: dict[str, User] = {}

    def register(self, email: str) -> User:
        canonical = normalize_email(email)
        if canonical in self._users:
            raise ValueError("email already registered")
        user = User(email=email)
        self._users[email] = user
        return user

    def find(self, email: str) -> User | None:
        return self._users.get(normalize_email(email))

