"""Registration and sign-in rules.

The two differ on case deliberately: sign-in demands the exact username, while
registration refuses a name already held in any casing.
"""

from __future__ import annotations

import uuid

import pytest

from cae import auth


def fresh_name() -> str:
    return f"Case{uuid.uuid4().hex[:8]}"


def test_signing_in_is_case_sensitive() -> None:
    name = fresh_name()
    auth.register(name, "correct-horse-battery")

    with pytest.raises(auth.AuthError, match="Incorrect username or password"):
        auth.login(name.upper(), "correct-horse-battery")
    with pytest.raises(auth.AuthError, match="Incorrect username or password"):
        auth.login(name.lower(), "correct-horse-battery")

    assert auth.login(name, "correct-horse-battery")["username"] == name


def test_the_wrong_case_is_refused_as_an_unknown_user_not_a_bad_password() -> None:
    """The message must not reveal that the name exists in another casing."""
    name = fresh_name()
    auth.register(name, "correct-horse-battery")

    with pytest.raises(auth.AuthError) as caught:
        auth.login(name.upper(), "correct-horse-battery")
    assert "Incorrect username or password." == str(caught.value)


def test_surrounding_whitespace_is_still_forgiven() -> None:
    name = fresh_name()
    auth.register(name, "correct-horse-battery")

    assert auth.login(f"  {name} ", "correct-horse-battery")["username"] == name


def test_a_name_cannot_be_reregistered_in_another_casing() -> None:
    name = fresh_name()
    auth.register(name, "correct-horse-battery")

    for variant in (name.upper(), name.lower(), name):
        with pytest.raises(auth.AuthError, match="already taken"):
            auth.register(variant, "another-valid-password")
