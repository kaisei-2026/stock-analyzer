"""読みやすいライトテーマ（高コントラスト）"""

import streamlit as st

# チャート色 — 緑＝陽線、赤＝陰線（国内慣行に合わせる設定も可能）
COLORS = {
    "bg": "#ffffff",
    "panel": "#f8fafc",
    "grid": "#cbd5e1",
    "text": "#0f172a",
    "muted": "#475569",
    "up": "#15803d",
    "down": "#b91c1c",
    "channel_hi": "#6d28d9",
    "channel_lo": "#c2410c",
    "channel_fill": "rgba(109, 40, 217, 0.06)",
    "buy": "#15803d",
    "sell": "#b91c1c",
    "equity": "#0369a1",
    "benchmark": "#64748b",
    "accent": "#1d4ed8",
}

CHART_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["panel"],
    font=dict(color=COLORS["text"], size=13, family="Segoe UI, Meiryo, Hiragino Sans, sans-serif"),
    xaxis=dict(
        gridcolor=COLORS["grid"],
        linecolor=COLORS["grid"],
        tickfont=dict(color=COLORS["muted"]),
        title_font=dict(color=COLORS["text"]),
    ),
    yaxis=dict(
        gridcolor=COLORS["grid"],
        linecolor=COLORS["grid"],
        tickfont=dict(color=COLORS["muted"]),
        title_font=dict(color=COLORS["text"]),
    ),
)


def inject_theme() -> None:
    c = COLORS
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #f1f5f9;
            color: {c["text"]};
        }}
        [data-testid="stSidebar"] {{
            background-color: #ffffff;
            border-right: 1px solid {c["grid"]};
        }}
        [data-testid="stSidebar"] * {{
            color: {c["text"]} !important;
        }}
        h1, h2, h3, h4, p, label, span {{
            color: {c["text"]};
        }}
        h1 {{
            color: {c["accent"]} !important;
            -webkit-text-fill-color: {c["accent"]} !important;
            font-weight: 700 !important;
        }}
        [data-testid="stMetric"] {{
            background: #ffffff;
            border: 1px solid {c["grid"]};
            border-radius: 8px;
            padding: 12px 16px;
            box-shadow: 0 1px 3px rgba(15,23,42,0.08);
        }}
        [data-testid="stMetricLabel"] {{
            color: {c["muted"]} !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {c["text"]} !important;
        }}
        .signal-banner {{
            padding: 14px 18px;
            border-radius: 8px;
            border: 1px solid {c["grid"]};
            background: #ffffff;
            margin-bottom: 1rem;
            color: {c["text"]};
        }}
        .signal-banner strong {{
            color: {c["accent"]};
        }}
        .workflow-ok {{ color: #15803d; font-weight: 600; }}
        .workflow-ng {{ color: #64748b; }}
        div[data-testid="stExpander"] {{
            background: #ffffff;
            border: 1px solid {c["grid"]};
            border-radius: 8px;
        }}
        [data-testid="stDataFrame"] {{
            border: 1px solid {c["grid"]};
            border-radius: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_figure(fig, height: int = 520):
    fig.update_layout(**CHART_LAYOUT, height=height, hovermode="x unified")
    return fig
