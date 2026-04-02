import phoenix as px
from openinference.instrumentation import OITracer

def start_phoenix():
    """Launch Phoenix dashboard and return a tracer."""
    session = px.launch_app()
    print(f"Phoenix UI: {session.url}")
    return session
