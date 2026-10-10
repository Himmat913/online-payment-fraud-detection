"""HTML builders for the parts of the page Streamlit widgets can't express.

Everything is rendered with st.html (no Markdown processing). Any value that
comes from data, the model or the user is escaped.
"""

from __future__ import annotations

from html import escape
from pathlib import Path


def load_css(path: Path) -> str:
    return f"<style data-fr-styles>{path.read_text(encoding='utf-8')}</style>"


def nav_current(title: str) -> str:
    return f'<span class="fr-nav-current" aria-current="page">{escape(title)}</span>'


def fmt_int(n) -> str:
    return f"{int(round(n)):,}"


def fmt_amount(n) -> str:
    return f"{n:,.2f}"


def fmt_pct(x, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def emphasis(before: str, italic: str, after: str = "") -> str:
    """Playfair headline with the template's bold-italic emphasis on one phrase."""
    return f"{escape(before)}<em>{escape(italic)}</em>{escape(after)}"


def page_header(label: str, title_html: str, intro: str = "") -> str:
    intro_html = f'<p class="fr-lede">{intro}</p>' if intro else ""
    return (
        '<header class="fr-page-header">'
        f'<p class="fr-label">{escape(label)}</p>'
        f"<h1>{title_html}</h1>{intro_html}"
        "</header>"
    )


def hero_copy() -> str:
    return (
        '<div class="fr-hero__copy">'
        '<p class="fr-label">Online payment fraud, PaySim simulation data</p>'
        '<h1 id="fr-hero-title">Find the <em>few</em> risky payments hiding in thousands of normal ones</h1>'
        '<p class="fr-hero__text">Explore where fraud concentrates, triage transactions with the '
        "trained neural network, and see what its scores mean at a real-world fraud rate.</p>"
        "</div>"
    )


def hero_art(stats: dict) -> str:
    tags = (
        f'<span class="fr-tag fr-tag--a">{fmt_int(stats["rows"])} transactions</span>'
        f'<span class="fr-tag fr-tag--b">{fmt_int(stats["fraud"])} confirmed fraud</span>'
        f'<span class="fr-tag fr-tag--c">{fmt_pct(stats["fraud_rate"], 2)} fraud rate</span>'
    )
    return (
        '<div class="fr-hero__art" aria-hidden="true">'
        '<div class="fr-hero__frame"></div>'
        '<div class="fr-hero__frame fr-hero__frame--inner"></div>'
        f'<div class="fr-hero__tags">{tags}</div>'
        "</div>"
    )


def numbers_row(items: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="fr-number"><p class="fr-number__value">{value}</p>'
        f'<p class="fr-number__text">{text}</p></div>'
        for value, text in items
    )
    return f'<section class="fr-numbers" aria-label="Key figures">{cells}</section>'


def section_heading(label: str, title_html: str, text: str = "") -> str:
    text_html = f'<p class="fr-muted">{text}</p>' if text else ""
    return (
        '<div class="fr-section-heading">'
        f'<p class="fr-label">{escape(label)}</p><h2>{title_html}</h2>{text_html}'
        "</div>"
    )


def card_title(title: str, text: str = "") -> str:
    text_html = f'<p class="fr-muted fr-small">{text}</p>' if text else ""
    return f'<div class="fr-card-title"><h3>{escape(title)}</h3>{text_html}</div>'


def note(text_html: str, tone: str = "plain") -> str:
    return f'<p class="fr-note fr-note--{tone}">{text_html}</p>'


def source_notice(source: str, rows: int, detail: str | None) -> str:
    if source == "mysql":
        text = f"Live data from the MySQL <code>transactions</code> table · {fmt_int(rows)} rows."
    else:
        text = (
            f"Bundled sample · {fmt_int(rows)} rows drawn from the 6.36M-row dataset "
            "after preprocessing."
        )
    detail_html = f'<br><span class="fr-small">{escape(detail)} Showing the bundled sample instead.</span>' if detail else ""
    return f'<p class="fr-source" role="status">{text}{detail_html}</p>'


def insight_list(items: list[tuple[str, str]]) -> str:
    rows = "".join(
        f'<li><p class="fr-insight__fact">{fact}</p><p class="fr-muted">{why}</p></li>'
        for fact, why in items
    )
    return f'<ul class="fr-insights">{rows}</ul>'


def score_result(score: float, threshold: float, flagged: bool, context: dict | None) -> str:
    context_html = ""
    if context:
        context_html = (
            '<p class="fr-result__context">At this threshold the model flags '
            f'<strong>{fmt_pct(context["flagged_share"])}</strong> of the dataset and catches '
            f'<strong>{context["fraud_caught"]} of {context["total_fraud"]}</strong> known fraud cases. '
            f'<strong>{context["precision_text"]}</strong> of flagged transactions are fraud.</p>'
        )
    verdict = "Flag for review" if flagged else "Let it through"
    summary = (
        "The score is above your threshold, so this transaction would be sent to an analyst."
        if flagged
        else "The score is at or below your threshold, so this transaction would not raise an alert."
    )
    tone = "flag" if flagged else "pass"
    return (
        f'<section class="fr-result fr-result--{tone}" role="status" aria-live="polite">'
        '<p class="fr-label">Model decision</p>'
        f'<p class="fr-result__verdict">{verdict}</p>'
        f'<p class="fr-result__summary">{summary}</p>'
        '<div class="fr-result__score">'
        f'<span class="fr-result__value">{score:.3f}</span>'
        f'<span class="fr-result__vs">score · threshold {threshold:.2f}</span>'
        "</div>"
        f'<div class="fr-meter" role="img" aria-label="Score {score:.3f}, threshold {threshold:.2f}">'
        f'<span class="fr-meter__fill" style="width:{score * 100:.1f}%"></span>'
        f'<span class="fr-meter__mark" style="left:{threshold * 100:.1f}%"></span></div>'
        f"{context_html}"
        "</section>"
    )


def empty_result() -> str:
    return (
        '<section class="fr-result fr-result--empty">'
        '<p class="fr-label">Model decision</p>'
        '<p class="fr-result__verdict">Waiting for a transaction</p>'
        '<p class="fr-result__summary">Fill in the details and select <strong>Score transaction</strong>. '
        "The decision, the raw score and what that threshold means in practice will appear here.</p>"
        "</section>"
    )


def spec_list(rows: list[tuple[str, str]]) -> str:
    items = "".join(f"<dt>{escape(k)}</dt><dd>{v}</dd>" for k, v in rows)
    return f'<dl class="fr-specs">{items}</dl>'


def steps(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<li class="fr-step"><span class="fr-step__n">{n}</span>'
        f'<p class="fr-step__title">{escape(title)}</p><p class="fr-muted">{text}</p></li>'
        for n, title, text in items
    )
    return f'<ol class="fr-steps">{cells}</ol>'


def confusion_matrix(cm: dict) -> str:
    def cell(value, label, css):
        return f'<div class="fr-cm__cell {css}"><span>{fmt_int(value)}</span><small>{label}</small></div>'

    return (
        '<div class="fr-cm" role="table" aria-label="Confusion matrix on the notebook test set">'
        '<div class="fr-cm__corner"></div>'
        '<div class="fr-cm__head">Predicted legitimate</div><div class="fr-cm__head">Predicted fraud</div>'
        '<div class="fr-cm__side">Actually legitimate</div>'
        f'{cell(cm["true_negative"], "correctly passed", "fr-cm__cell--ok")}'
        f'{cell(cm["false_positive"], "false alarms", "fr-cm__cell--warn")}'
        '<div class="fr-cm__side">Actually fraud</div>'
        f'{cell(cm["false_negative"], "missed fraud", "fr-cm__cell--warn")}'
        f'{cell(cm["true_positive"], "fraud caught", "fr-cm__cell--ok")}'
        "</div>"
    )


def callout(title_html: str, text_html: str) -> str:
    return f'<div class="fr-callout"><h2>{title_html}</h2><p>{text_html}</p></div>'


def footer(source: str) -> str:
    source_text = "MySQL transactions table" if source == "mysql" else "bundled 50,000-row sample"
    return (
        '<footer class="fr-footer">'
        '<div class="fr-footer__brand">'
        '<p class="fr-wordmark">fraud risk analytics</p>'
        f'<p>Data: {source_text}.<br>Model: dense neural network trained on PaySim.</p>'
        "</div>"
        '<div class="fr-footer__card">'
        '<p class="fr-footer__title">Built for <em>analysis</em>, not for blocking payments</p>'
        "<p>PaySim is simulated mobile-money data. Scores here are research outputs: "
        "use them to explore patterns and prioritise review, never as the sole basis for "
        "declining a real customer’s payment.</p>"
        "</div>"
        "</footer>"
    )
