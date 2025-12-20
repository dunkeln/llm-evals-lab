import marimo

__generated_with = "0.18.3"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import duckdb as db
    import altair as alt
    import polars as pl

    conn = db.connect("../warehouse/llm_evals.duckdb")
    df = conn.sql("select * from analytics.models_reasoning_scores").df()
    return df, mo


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, mo):
    metrics = mo.sql(
        f"""
        with metrics_table as (
        	select id, metadata, provider, run_timestamp, model, input_tokens, output_tokens, total_tokens, latency, temperature, max_tokens, top_p, cosine_similarity, token_f1
        	from df
        ) select * from metrics_table order by metadata, provider;
        """
    )
    return (metrics,)


@app.cell
def _(metrics, mo):
    _df = mo.sql(
        f"""
        select metadata, avg(total_tokens), avg(cosine_similarity), avg(latency), avg(temperature),  from metrics group by provider;
        """
    )
    return


if __name__ == "__main__":
    app.run()
