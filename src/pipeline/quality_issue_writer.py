# src/pipeline/quality_issue_writer.py

from src.pipeline.db import get_sql_conn


def write_quality_issues(validation):
    """
    Writes warnings/errors from Silver validation into dbo.QualityIssues.
    """

    conn = get_sql_conn()
    cur = conn.cursor()

    for msg in validation.errors:
        cur.execute(
            """
            INSERT INTO dbo.QualityIssues (DocId, ClientId, TaxYear, IssueType, Message)
            VALUES (?, ?, ?, 'error', ?)
            """,
            (validation.doc_id, validation.client_id, validation.tax_year, msg),
        )

    for msg in validation.warnings:
        cur.execute(
            """
            INSERT INTO dbo.QualityIssues (DocId, ClientId, TaxYear, IssueType, Message)
            VALUES (?, ?, ?, 'warning', ?)
            """,
            (validation.doc_id, validation.client_id, validation.tax_year, msg),
        )

    conn.commit()
    conn.close()
