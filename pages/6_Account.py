"""Account settings: changing the password."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from cae import auth  # noqa: E402
from cae.ui.shell import (  # noqa: E402
    page_header,
    page_setup,
    require_login,
    sidebar_account,
)

page_setup("Account", icon="👤")
user = require_login()
sidebar_account(user)

page_header("Account")

st.markdown("### Change your password")
with st.form("change_password"):
    current = st.text_input("Current password", type="password")
    replacement = st.text_input("New password", type="password")
    confirm = st.text_input("Confirm new password", type="password")
    submitted = st.form_submit_button("Update password")
if submitted:
    if replacement != confirm:
        st.error("The two new passwords do not match.")
    else:
        try:
            auth.change_password(user["id"], current, replacement)
            st.success("Password updated.")
        except auth.AuthError as exc:
            st.error(str(exc))
