"""Unit tests for core.visualizer"""

import pandas as pd
import pytest
from core.visualizer import classify_columns, recommend_chart_type, render_plotly_chart


def test_classify_columns():
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "category": ["A", "B", "C"],
        "value": [10.5, 20.0, 30.2]
    })
    classified = classify_columns(df)
    assert "value" in classified["numeric"]
    assert "category" in classified["categorical"]
    assert "date" in classified["datetime"]


def test_recommend_chart_type():
    df_bar = pd.DataFrame({
        "department": ["Eng", "Mkt", "Sales"],
        "salary": [100000, 80000, 75000]
    })
    chart_type, x, y = recommend_chart_type(df_bar)
    assert chart_type == "bar"
    assert x == "department"
    assert y == "salary"


def test_render_plotly_chart():
    df = pd.DataFrame({
        "dept": ["Eng", "Sales"],
        "revenue": [500000, 300000]
    })
    fig = render_plotly_chart(df, chart_type="bar", x_col="dept", y_col="revenue")
    assert fig is not None
    assert len(fig.data) >= 1
