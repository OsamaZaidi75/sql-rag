"""
Dynamic Visualization Engine for SQL RAG
Uses Plotly for interactive, publication-ready charts with automatic
column detection and intelligent chart recommendation.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, Dict, Any, List


def classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Classifies dataframe columns into categorical, numeric, and datetime types."""
    numeric_cols = []
    datetime_cols = []
    categorical_cols = []

    for col in df.columns:
        # Check datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
            continue

        # Try parsing string to date if it looks like date
        # (is_string_dtype covers both pandas 2.x `object` and 3.x `str` dtypes)
        if pd.api.types.is_string_dtype(df[col]) and len(df) > 0:
            sample_val = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else ""
            if any(char in sample_val for char in ["-", "/"]) and len(sample_val) in (7, 10, 19, 23):
                try:
                    pd.to_datetime(df[col].dropna().head(5))
                    datetime_cols.append(col)
                    continue
                except Exception:
                    pass

        # Check numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "datetime": datetime_cols,
        "categorical": categorical_cols,
    }


def recommend_chart_type(df: pd.DataFrame) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Intelligently recommends the best chart type and default X/Y axis based on data characteristics.
    Returns (chart_type, recommended_x, recommended_y).
    """
    if df.empty or len(df.columns) < 1:
        return "table", None, None

    cols = classify_columns(df)
    num_cols = cols["numeric"]
    cat_cols = cols["categorical"]
    date_cols = cols["datetime"]

    # Case 1: Date + Numeric -> Line chart
    if date_cols and num_cols:
        return "line", date_cols[0], num_cols[0]

    # Case 2: Categorical + Numeric -> Bar chart
    if cat_cols and num_cols:
        # If <= 6 categories, Pie or Donut might also be great, default to Bar
        return "bar", cat_cols[0], num_cols[0]

    # Case 3: 2+ Numeric columns -> Scatter or Line
    if len(num_cols) >= 2:
        return "scatter", num_cols[0], num_cols[1]

    # Case 4: Single Numeric column -> Histogram
    if len(num_cols) == 1:
        return "histogram", num_cols[0], None

    # Case 5: Only Categorical -> Bar chart of value counts
    if cat_cols:
        return "bar_counts", cat_cols[0], None

    return "table", df.columns[0], df.columns[1] if len(df.columns) > 1 else None


def render_plotly_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: Optional[str] = None,
    color_col: Optional[str] = None,
    title: Optional[str] = None,
    theme: str = "plotly_white"
) -> Optional[go.Figure]:
    """Generates a styled Plotly figure."""
    if df.empty or not x_col or x_col not in df.columns:
        return None

    try:
        plot_df = df.copy()

        # Clean NaNs in selected columns
        cols_to_clean = [x_col]
        if y_col and y_col in plot_df.columns:
            cols_to_clean.append(y_col)
        if color_col and color_col in plot_df.columns:
            cols_to_clean.append(color_col)
        plot_df = plot_df.dropna(subset=cols_to_clean)

        chart_type = chart_type.lower()

        if chart_type == "bar":
            if y_col and y_col in plot_df.columns:
                fig = px.bar(
                    plot_df,
                    x=x_col,
                    y=y_col,
                    color=color_col if color_col in plot_df.columns else None,
                    title=title or f"{y_col} by {x_col}",
                    template=theme,
                    text_auto=".2s"
                )
            else:
                fig = px.bar(
                    plot_df[x_col].value_counts().reset_index(),
                    x=x_col,
                    y="count",
                    title=title or f"Count by {x_col}",
                    template=theme
                )

        elif chart_type == "line":
            fig = px.line(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_col if color_col in plot_df.columns else None,
                markers=True,
                title=title or f"{y_col} over {x_col}",
                template=theme
            )

        elif chart_type == "area":
            fig = px.area(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_col if color_col in plot_df.columns else None,
                title=title or f"{y_col} Area over {x_col}",
                template=theme
            )

        elif chart_type in ("pie", "donut"):
            hole = 0.4 if chart_type == "donut" else 0.0
            if y_col and y_col in plot_df.columns:
                fig = px.pie(
                    plot_df,
                    names=x_col,
                    values=y_col,
                    hole=hole,
                    title=title or f"{y_col} Distribution by {x_col}",
                    template=theme
                )
            else:
                counts = plot_df[x_col].value_counts().reset_index()
                fig = px.pie(
                    counts,
                    names=x_col,
                    values="count",
                    hole=hole,
                    title=title or f"Distribution of {x_col}",
                    template=theme
                )

        elif chart_type == "scatter":
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_col if color_col in plot_df.columns else None,
                trendline="ols" if len(plot_df) > 3 and pd.api.types.is_numeric_dtype(plot_df[x_col]) and pd.api.types.is_numeric_dtype(plot_df[y_col]) else None,
                title=title or f"{y_col} vs {x_col}",
                template=theme
            )

        elif chart_type == "histogram":
            fig = px.histogram(
                plot_df,
                x=x_col,
                color=color_col if color_col in plot_df.columns else None,
                marginal="box",
                title=title or f"Distribution of {x_col}",
                template=theme
            )

        else:
            return None

        fig.update_layout(
            margin=dict(l=30, r=30, t=50, b=30),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    except Exception:
        return None
