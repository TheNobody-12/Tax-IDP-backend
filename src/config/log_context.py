"""Thread-local log context for correlation IDs."""
import threading

_tls = threading.local()

def set_log_context(doc_id: str | None = None, job_id: str | None = None):
    if doc_id is not None:
        _tls.doc_id = doc_id
    if job_id is not None:
        _tls.job_id = job_id


def get_log_context():
    return {
        "doc_id": getattr(_tls, "doc_id", None),
        "job_id": getattr(_tls, "job_id", None),
    }
