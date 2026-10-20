"""User registration and login with bcrypt-hashed passwords.

Users are returned as plain dicts so they can be stored in ``st.session_state``.
"""

from __future__ import annotations

import re
from typing import Any

import bcrypt

from cae.db import repo
from cae.db.models import User
from cae.db.session import session_scope

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_BYTES = 72


class AuthError(Exception):
    """Registration or login failed for a reason worth showing the user."""


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


_DUMMY_HASH = hash_password("placeholder-value")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def _as_dict(user) -> dict[str, Any]:
    return {"id": user.id, "username": user.username}


def validate_credentials(username: str, password: str) -> None:
    if not USERNAME_RE.match(username or ""):
        raise AuthError(
            "Usernames must be 3-32 characters long and may contain only letters, "
            "digits, dots, hyphens and underscores."
        )
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Passwords must be at least {MIN_PASSWORD_LENGTH} characters long.")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise AuthError("That password is too long. Please use 72 bytes or fewer.")


def register(username: str, password: str) -> dict[str, Any]:
    username = (username or "").strip()
    validate_credentials(username, password)
    with session_scope() as session:
        if repo.username_is_taken(session, username):
            raise AuthError("That username is already taken.")
        user = repo.create_user(
            session, username=username, password_hash=hash_password(password)
        )
        return _as_dict(user)


def change_password(user_id: int, current_password: str, new_password: str) -> None:
    with session_scope() as session:
        user = session.get(User, user_id)
        if user is None:
            raise AuthError("That account no longer exists.")
        if not verify_password(current_password or "", user.password_hash):
            raise AuthError("Your current password is not correct.")
        validate_credentials(user.username, new_password)
        user.password_hash = hash_password(new_password)


def login(username: str, password: str) -> dict[str, Any]:
    with session_scope() as session:
        user = repo.get_user_by_username(session, (username or "").strip())
        stored = user.password_hash if user else _DUMMY_HASH
        if not verify_password(password or "", stored) or user is None:
            raise AuthError("Incorrect username or password.")
        repo.touch_login(session, user.id)
        return _as_dict(user)
