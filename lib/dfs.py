import logging
from typing import List, Union
import polars as pl
import pandas as pd
import polars.selectors as cs
from typing import Sequence
import polars.datatypes as datatypes


@pl.api.register_dataframe_namespace("custom")
@pl.api.register_lazyframe_namespace("custom")
class CustomFrame:
    def __init__(self, df: pl.DataFrame | pl.LazyFrame):
        if isinstance(df, pl.DataFrame):
            self._df = df.lazy()
            self._was_df = True
        else:
            self._df = df
            self._was_df = False


# https://docs.pola.rs/api/python/stable/reference/api.html
@pl.api.register_expr_namespace("custom")
class CustomExpr:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr


def add_to_class(cls):
    """Decorator to add a function as a method to the given class."""

    def decorator(func):
        setattr(cls, func.__name__, func)
        return func

    return decorator


def drop_nulls(df):
    not_null_cols = filter(lambda x: x.null_count() != df.height, df)
    not_null_col_names = map(lambda x: x.name, not_null_cols)
    return df.select(not_null_col_names)


def preview_df(frame, head=True, glimpse=True, tail=True, describe=True):
    if isinstance(frame, (pl.DataFrame)):  # Check for polars DataFrame
        print("-" * 50)
        if head:
            print(frame)
        if glimpse:
            print("\nGlimpse:")
            frame.glimpse()

        print("\nDescription:")
        print(frame.describe())

        print("\nTail:")
        print(frame.tail())

        print("-" * 50)
    elif isinstance(frame, (pd.DataFrame)):
        print("-" * 50)
        if head:
            print(frame)
        if glimpse:
            print("\nGlimpse:")
            print(frame.info())

        print("\nDescription:")
        print(frame.describe())

        print("\nTail:")
        print(frame.tail())

        print("-" * 50)


def process_categoricals(
    df: Union[pl.DataFrame, pd.DataFrame],
    categorical_cols: List[str] = None,
    prefix_sep: str = "_",
    impute="mode",
    drop_first=True,
) -> pl.DataFrame:
    """
    Polars version of impute_and_encode function.
    Mimics sklearn's SimpleImputer(strategy="most_frequent") + OneHotEncoder(handle_unknown="ignore")

    Args:
        df: Input DataFrame (Polars or Pandas)
        prefix_sep: Separator between column name and category in new column names
        impute: Impute expr, default: mode

    Returns:
        Polars DataFrame with imputed and encoded categorical columns
    """

    if not categorical_cols:
        categorical_cols = df.select(cs.categorical() | cs.string()).columns
        logging.info("Guessing Categorical columns: ", categorical_cols)

    if impute == "mode":
        impute = [
            pl.col(col).fill_null(pl.col(col).mode().first())
            for col in categorical_cols
        ]

    # Apply imputation
    processed_df = df.with_columns(impute)

    processed_df = processed_df.to_dummies(
        categorical_cols, separator=prefix_sep, drop_first=drop_first
    )

    return processed_df


def process_nulls():
    return


@add_to_class(CustomFrame)
def dropna(
    self: pl.LazyFrame,
    how: str = "any",
    thresh: int = None,
    subset: str | Sequence[str] = None,
) -> pl.LazyFrame:
    """
    Remove null and NaN values https://stackoverflow.com/questions/73971106/polars-dropna-equivalent-on-list-of-columns
    """
    df = self._df

    if subset is None:
        subset = cs.all()
    else:
        subset = cs.by_name(subset)

    num_subset = cs.numeric() & subset

    # todo: fix
    expr = subset.is_not_null() & num_subset.is_not_nan()

    if thresh is not None:
        result = df.filter(pl.sum_horizontal(expr) >= thresh)
    elif how == "any":
        result = df.filter(pl.all_horizontal(expr))
    elif how == "all":
        result = df.filter(pl.any_horizontal(expr))
    else:
        ...

    result = df._from_pyldf(result._ldf)

    return result.collect() if self._was_df else result


def drop_rows(
    df: pl.DataFrame,
    threshold: float | int = 1,
    col_expr=None,
    count_null=lambda row: sum(1 for x in row.values() if x is None),
) -> pl.DataFrame:
    col_expr = pl.all() if col_expr is None else col_expr

    threshold = (
        threshold
        if isinstance(threshold, int)
        else (df.select(col_expr).width * threshold)
    )

    df = df.filter(
        pl.sum_horizontal(
            pl.struct(col_expr).map_elements(count_null, return_dtype=pl.Int64)
        )
        < threshold
    )
    return df


def drop_cols(
    df, col_expr=None, drop_criterion=lambda col: any(e is not None for e in col)
):
    col_expr = pl.all() if col_expr is None else col_expr

    return df.select(
        col.name
        for col in df.select(
            col_expr.map_batches(lambda col: pl.Series([not drop_criterion(col)]))
        )
        if col.all()
    )

@add_to_class(CustomExpr)
def clean_numeric(self: pl.Expr) -> pl.Expr:
    """Converts string to numeric column by dropping non-numeric characters

    Returns:
        pl.Expr

    Example:
        df.with_columns(
            pl.col("Fiber_Diameter_(µm)").my.clean_numeric()
        )
    """
    return (
        self._expr.str.replace_all(
            r"[^0-9.]", ""
        ).cast(  # Replace non-digit and non-dot characters
            pl.Float64
        )  # Cast to Float64
    )

@add_to_class(CustomExpr)
def clean_str(self: pl.Expr) -> pl.Expr:
    """Converts all strings to lowercase and removes all special characters.

    Returns:
        pl.Expr

    Example:
        df.with_columns(
            pl.col("some_column").my.clean_str()
        )
    """
    return (
        self._expr.str.to_lowercase()
    ).str.replace_all(  # Convert all characters to lowercase
        r"[^a-z0-9\s]", ""
    )