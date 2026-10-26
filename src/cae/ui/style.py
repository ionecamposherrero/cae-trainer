"""The app's look: theme-aware CSS, the logo and a few HTML building blocks.

Colours and fonts for Streamlit's own widgets live in ``.streamlit/config.toml``;
this module styles what the theme cannot reach, keyed containers among them. A
container whose key starts with ``cae_card_`` is drawn as a card, ``cae_panel_``
as a flat panel, ``cae_paper_`` as a sheet of the question paper, and
``cae_hero`` as the banner at the top of the home page.
"""

from __future__ import annotations

import html
from typing import Iterable

import streamlit as st

SERIF = "'Source Serif 4', Georgia, 'Times New Roman', serif"
GRADIENT = "linear-gradient(135deg, #6366f1 0%, #7c3aed 100%)"

_LIGHT = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "primary": "#4f46e5",
    "primary_ink": "#4338ca",
    "tint": "#eef0fe",
    "glow": "rgba(99, 102, 241, 0.10)",
    "card": "#ffffff",
    "paper": "#ffffff",
    "border": "#e3e6ef",
    "shadow": "rgba(15, 23, 42, 0.05)",
    "shadow_strong": "rgba(15, 23, 42, 0.16)",
    "wordmark": "#0f172a",
    "tagline": "#7c3aed",
}
_DARK = {
    "ink": "#e5e7eb",
    "muted": "#94a3b8",
    "primary": "#818cf8",
    "primary_ink": "#c7d2fe",
    "tint": "#1e2340",
    "glow": "rgba(99, 102, 241, 0.16)",
    "card": "#121826",
    "paper": "#121826",
    "border": "#232a3d",
    "shadow": "rgba(0, 0, 0, 0.30)",
    "shadow_strong": "rgba(0, 0, 0, 0.65)",
    "wordmark": "#f1f5f9",
    "tagline": "#a5b4fc",
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&display=swap');

:root {
    --cae-ink: <ink>;
    --cae-muted: <muted>;
    --cae-primary: <primary>;
    --cae-primary-ink: <primary_ink>;
    --cae-tint: <tint>;
    --cae-card: <card>;
    --cae-paper: <paper>;
    --cae-border: <border>;
    --cae-shadow: <shadow>;
    --cae-shadow-strong: <shadow_strong>;
    --cae-gradient: <gradient>;
    --cae-serif: <serif>;
    --cae-elevation: 0 1px 2px var(--cae-shadow), 0 10px 30px -18px var(--cae-shadow-strong);
}

[data-testid="stSidebarNavItems"] li:first-child [data-testid="stSidebarNavLink"] span {
    display: none;
}
[data-testid="stSidebarNavItems"] li:first-child [data-testid="stSidebarNavLink"]::after {
    content: "Home";
}

[data-testid="stApp"] {
    background-image:
        radial-gradient(900px 480px at 100% -8%, <glow>, transparent 65%),
        radial-gradient(700px 420px at -10% 110%, <glow>, transparent 60%);
    background-attachment: fixed;
}
.block-container {
    max-width: 1200px;
    padding-top: 2.4rem;
    padding-bottom: 4rem;
}
[data-testid="stHeader"] { background: transparent; }
h1 { font-weight: 800 !important; letter-spacing: -0.025em; }
h2, h3 { font-weight: 700 !important; letter-spacing: -0.015em; }
h4, h5 { font-weight: 700 !important; letter-spacing: -0.005em; }

[data-testid="stBaseButton-primary"]:not(:disabled),
[data-testid="stBaseButton-primaryFormSubmit"]:not(:disabled) {
    background-image: var(--cae-gradient);
    border-color: transparent;
    box-shadow: 0 8px 20px -10px rgba(79, 70, 229, 0.75);
    transition: transform 0.15s ease, box-shadow 0.15s ease, filter 0.15s ease;
}
[data-testid="stBaseButton-primary"]:not(:disabled):hover,
[data-testid="stBaseButton-primaryFormSubmit"]:not(:disabled):hover {
    transform: translateY(-1px);
    filter: brightness(1.06);
    box-shadow: 0 12px 24px -10px rgba(79, 70, 229, 0.85);
}

.cae-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.28rem 0.75rem;
    border-radius: 999px;
    background: var(--cae-tint);
    color: var(--cae-primary-ink);
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    margin-bottom: -0.5rem;
}
.cae-eyebrow::before {
    content: "";
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--cae-gradient);
}
.cae-lead {
    font-size: 1.06rem;
    line-height: 1.65;
    color: var(--cae-muted);
    max-width: 72ch;
    margin: -0.3rem 0 0.8rem 0;
}

.cae-chips { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.3rem 0 0.4rem 0; }
.cae-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    background: var(--cae-tint);
    color: var(--cae-primary-ink);
    font-size: 0.8rem;
    font-weight: 600;
}

.cae-steps { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0 0 0.6rem 0; }
.cae-step {
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    border: 1px solid var(--cae-border);
    background: var(--cae-card);
    color: var(--cae-muted);
    font-size: 0.78rem;
    font-weight: 600;
}
.cae-step.is-done {
    background: var(--cae-tint);
    border-color: transparent;
    color: var(--cae-primary-ink);
}
.cae-step.is-current {
    background: var(--cae-gradient);
    border-color: transparent;
    color: #ffffff;
    box-shadow: 0 6px 16px -8px rgba(79, 70, 229, 0.8);
}

[data-testid="stMetric"] {
    background: var(--cae-card);
    border: 1px solid var(--cae-border);
    border-radius: 18px;
    padding: 1rem 1.2rem;
    box-shadow: var(--cae-elevation);
}
[data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -0.03em; }
[data-testid="stMetricLabel"] p {
    font-size: 0.84rem;
    font-weight: 500;
    color: var(--cae-muted);
}

.st-key-cae_hero {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 90% 0%, rgba(56, 189, 248, 0.45), transparent 45%),
        radial-gradient(circle at 0% 100%, rgba(236, 72, 153, 0.30), transparent 45%),
        linear-gradient(135deg, #4338ca 0%, #6d28d9 60%, #7c3aed 100%);
    border-radius: 28px;
    padding: 2.6rem 2.8rem 2.3rem 2.8rem;
    box-shadow: 0 30px 60px -32px rgba(67, 56, 202, 0.75);
}
.st-key-cae_hero::after {
    content: "";
    position: absolute;
    right: -80px;
    bottom: -120px;
    width: 320px;
    height: 320px;
    border-radius: 50%;
    border: 48px solid rgba(255, 255, 255, 0.07);
    pointer-events: none;
}
.st-key-cae_hero h1,
.st-key-cae_hero p,
.st-key-cae_hero li,
.st-key-cae_hero b,
.st-key-cae_hero strong { color: #ffffff !important; }
.st-key-cae_hero .cae-eyebrow {
    background: rgba(255, 255, 255, 0.16);
    color: #ffffff;
    backdrop-filter: blur(6px);
}
.st-key-cae_hero .cae-eyebrow::before { background: #a5f3fc; }
.st-key-cae_hero .cae-lead { color: rgba(255, 255, 255, 0.86) !important; }
.st-key-cae_hero .cae-chip {
    background: rgba(255, 255, 255, 0.14);
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.18);
}

.cae-features { list-style: none; padding: 0; margin: 0.4rem 0 0.2rem 0; }
.cae-features li {
    position: relative;
    padding: 0.42rem 0 0.42rem 2rem;
    line-height: 1.5;
    font-size: 0.97rem;
}
.cae-features li::before {
    content: "✓";
    position: absolute;
    left: 0;
    top: 0.5rem;
    width: 1.3rem;
    height: 1.3rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    color: #4338ca;
    background: #ffffff;
}

.cae-disclaimer {
    font-size: 0.78rem;
    color: var(--cae-muted);
    text-align: center;
    margin-top: 1.2rem;
}

[class*="st-key-cae_card_"] {
    background: var(--cae-card);
    border: 1px solid var(--cae-border);
    border-radius: 20px;
    padding: 1.35rem 1.35rem 1.05rem 1.35rem;
    box-shadow: var(--cae-elevation);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    min-height: 15.5rem;
}
[class*="st-key-cae_card_"]:hover {
    transform: translateY(-3px);
    border-color: var(--cae-primary);
    box-shadow: 0 2px 4px var(--cae-shadow), 0 22px 40px -22px var(--cae-shadow-strong);
}
[class*="st-key-cae_card_"] > div:last-child { margin-top: auto; }
[class*="st-key-cae_card_"] [data-testid="stHeaderActionElements"] { display: none; }
[class*="st-key-cae_card_"] h4 { padding: 0.15rem 0 0 0; font-size: 1.12rem; }
.cae-card-icon {
    width: 2.6rem;
    height: 2.6rem;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--cae-gradient);
    color: #ffffff;
    box-shadow: 0 10px 20px -10px rgba(79, 70, 229, 0.8);
    margin-bottom: 0.2rem;
}
.cae-card-icon svg { width: 1.3rem; height: 1.3rem; }
.cae-card-kicker {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--cae-primary);
    margin: 0.9rem 0 -0.25rem 0;
}
.cae-card-text {
    color: var(--cae-muted);
    font-size: 0.92rem;
    line-height: 1.55;
    margin: 0;
}
[class*="st-key-cae_card_"] [data-testid="stPageLink"] a {
    border-radius: 999px;
    background: var(--cae-tint);
    padding: 0.2rem 0.85rem;
    width: fit-content;
}
[class*="st-key-cae_card_"] [data-testid="stPageLink"] p {
    color: var(--cae-primary-ink);
    font-weight: 600;
}

[class*="st-key-cae_panel_"] {
    background: var(--cae-card);
    border: 1px solid var(--cae-border);
    border-radius: 20px;
    padding: 1.4rem 1.5rem;
    box-shadow: var(--cae-elevation);
}
[class*="st-key-cae_panel_"] [data-testid="stHeaderActionElements"] { display: none; }

[class*="st-key-cae_paper_"],
.cae-script {
    background: var(--cae-paper);
    border: 1px solid var(--cae-border);
    border-radius: 18px;
    padding: 1.7rem 2rem 1.3rem 2rem;
    box-shadow: var(--cae-elevation);
}
[class*="st-key-cae_paper_"] p,
[class*="st-key-cae_paper_"] li,
.cae-script {
    font-family: var(--cae-serif);
    font-size: 1.07rem;
    line-height: 1.9;
}
.cae-script { margin-bottom: 0.6rem; }
[class*="st-key-cae_paper_"] [data-testid="stHeaderActionElements"] { display: none; }

.cae-gap {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    min-width: 2rem;
    justify-content: center;
    padding: 0 0.6rem;
    margin: 0 0.15rem;
    border-radius: 999px;
    background: var(--cae-tint);
    color: var(--cae-primary-ink);
    box-shadow: inset 0 0 0 1.5px var(--cae-primary);
    font-family: Inter, sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    line-height: 1.6;
    vertical-align: 0.06em;
}
.cae-gap b { font-weight: 600; letter-spacing: 0.04em; }
.cae-keyword {
    display: inline-block;
    margin: 0.1rem 0 0.4rem 0;
    padding: 0.3rem 1rem;
    border-radius: 10px;
    background: var(--cae-tint);
    color: var(--cae-primary-ink);
    box-shadow: inset 0 0 0 1.5px var(--cae-primary);
    font-family: Inter, sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.12em;
}

.cae-user {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.7rem 0.8rem;
    border-radius: 16px;
    background: var(--cae-tint);
    margin-bottom: 0.6rem;
}
.cae-avatar {
    flex: 0 0 auto;
    width: 2.25rem;
    height: 2.25rem;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--cae-gradient);
    color: #ffffff;
    font-weight: 700;
    font-size: 1rem;
}
.cae-user-label { font-size: 0.72rem; color: var(--cae-muted); }
.cae-user-name { font-weight: 600; line-height: 1.25; word-break: break-all; }

@media (max-width: 640px) {
    .st-key-cae_hero { padding: 1.8rem 1.4rem 1.6rem 1.4rem; border-radius: 22px; }
    [class*="st-key-cae_paper_"], .cae-script { padding: 1.2rem 1.1rem; }
}

[data-testid="stExpander"] details {
    border-radius: 14px;
    background: var(--cae-card);
}
[data-testid="stChatMessage"] { border-radius: 18px; }
[data-testid="stAlert"] > div { border-radius: 14px; }
</style>
"""

_MARK = (
    '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#6366f1"/><stop offset="1" stop-color="#7c3aed"/>'
    "</linearGradient></defs>"
    '<rect x="2" y="2" width="40" height="40" rx="12" fill="url(#g)"/>'
    '<text x="22" y="28.5" text-anchor="middle" '
    "font-family=\"'Segoe UI', Helvetica, Arial, sans-serif\" font-size=\"17\" "
    'font-weight="800" fill="#ffffff">C1</text>'
)

LOGO_ICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="44" height="44" viewBox="0 0 44 44">'
    + _MARK
    + "</svg>"
)

_ICONS = {
    "book": '<path d="M2 4h6a4 4 0 0 1 4 4v13a3 3 0 0 0-3-3H2z"/>'
    '<path d="M22 4h-6a4 4 0 0 0-4 4v13a3 3 0 0 1 3-3h7z"/>',
    "pen": '<path d="M12 20h9"/>'
    '<path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
    "timer": '<circle cx="12" cy="13" r="8"/><path d="M12 9v4l2 2"/><path d="M9 2h6"/>',
    "chart": '<path d="M3 3v18h18"/><path d="m7 15 4-4 3 3 6-6"/>',
    "chat": '<path d="M21 12a8 8 0 0 1-11.6 7.1L3 21l1.9-6.4A8 8 0 1 1 21 12z"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
}


def logo(wordmark: str, tagline: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="220" height="44" '
        'viewBox="0 0 220 44">'
        + _MARK
        + '<text x="54" y="21" font-family="\'Segoe UI\', Helvetica, Arial, sans-serif" '
        f'font-size="17" font-weight="800" fill="{wordmark}">C1 Advanced</text>'
        '<text x="54" y="36" font-family="\'Segoe UI\', Helvetica, Arial, sans-serif" '
        f'font-size="11" font-weight="600" fill="{tagline}">Exam Trainer</text>'
        "</svg>"
    )


def tokens() -> dict[str, str]:
    """The colours stepped for whichever theme the viewer is using."""
    try:
        return _DARK if st.context.theme.type == "dark" else _LIGHT
    except Exception:
        return _LIGHT


def css() -> str:
    text = _CSS.replace("<serif>", SERIF).replace("<gradient>", GRADIENT)
    for name, value in tokens().items():
        text = text.replace(f"<{name}>", value)
    return text


def inject() -> None:
    st.markdown(css(), unsafe_allow_html=True)
    colours = tokens()
    st.logo(
        logo(colours["wordmark"], colours["tagline"]), size="large", icon_image=LOGO_ICON
    )


def eyebrow(text: str) -> None:
    st.markdown(
        f'<div class="cae-eyebrow">{html.escape(text)}</div>', unsafe_allow_html=True
    )


def lead(text: str) -> None:
    st.markdown(f'<p class="cae-lead">{html.escape(text)}</p>', unsafe_allow_html=True)


def chips(items: Iterable[str]) -> None:
    spans = "".join(f'<span class="cae-chip">{html.escape(item)}</span>' for item in items)
    st.markdown(f'<div class="cae-chips">{spans}</div>', unsafe_allow_html=True)


def steps(labels: list[str], *, current: int, done: set[int]) -> None:
    """Where the candidate is in the paper: the open step, and those confirmed."""
    spans = []
    for index, text in enumerate(labels):
        state = "is-current" if index == current else "is-done" if index in done else ""
        spans.append(f'<span class="cae-step {state}">{html.escape(text)}</span>')
    st.markdown(f'<div class="cae-steps">{"".join(spans)}</div>', unsafe_allow_html=True)


def features(items: Iterable[tuple[str, str]]) -> None:
    """A ticked list of ``(bold lead, rest of the line)`` pairs."""
    rows = "".join(
        f"<li><b>{html.escape(head)}</b> {html.escape(rest)}</li>" for head, rest in items
    )
    st.markdown(f'<ul class="cae-features">{rows}</ul>', unsafe_allow_html=True)


def user_badge(username: str) -> None:
    name = html.escape(username)
    initial = html.escape(username[:1].upper() or "?")
    st.markdown(
        f'<div class="cae-user"><div class="cae-avatar">{initial}</div>'
        f'<div><div class="cae-user-label">Signed in as</div>'
        f'<div class="cae-user-name">{name}</div></div></div>',
        unsafe_allow_html=True,
    )


def nav_card(
    key: str, page: str, *, kicker: str, title: str, icon: str, text: str, cta: str
) -> None:
    """A card that leads to another page. ``icon`` names one of ``_ICONS``."""
    glyph = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{_ICONS[icon]}</svg>'
    )
    with st.container(key=f"cae_card_{key}"):
        st.markdown(
            f'<div class="cae-card-icon">{glyph}</div>'
            f'<div class="cae-card-kicker">{html.escape(kicker)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"#### {title}")
        st.markdown(
            f'<p class="cae-card-text">{html.escape(text)}</p>', unsafe_allow_html=True
        )
        st.page_link(page, label=cta, icon=":material/arrow_forward:")


def disclaimer(text: str) -> None:
    st.markdown(
        f'<div class="cae-disclaimer">{html.escape(text)}</div>', unsafe_allow_html=True
    )
