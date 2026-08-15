# Store and index users by canonical email

Registration currently computes a canonical email but stores both the `User`
and dictionary entry with the raw input. This breaks lookup and duplicate
detection when casing or surrounding whitespace differs.

Acceptance requirements:

- Add a `User.from_email(email: str) -> User` class method in `benchmark/users.py`
  that stores `normalize_email(email)` in `User.email`.
- Make `UserService.register` construct users through that class method.
- Index registered users by their canonical email.
- Reject registrations that differ from an existing email only by casing or
  surrounding whitespace.
- Keep tests read-only and change only `benchmark/users.py` and
  `benchmark/user_service.py`.

