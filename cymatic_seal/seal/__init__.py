from .certificate import generate_certificate, SealCertificate


def seal_audio(*args, **kwargs):
    """Lazy wrapper — avoids importing torch at package-import time."""
    from .pipeline import seal_audio as _seal_audio
    return _seal_audio(*args, **kwargs)
