"""Compatibility entry point for Vertex-based retraining.

All normal retraining must run as a Vertex AI Custom Job. Keep this command
for existing course notes, but delegate directly to the authenticated Vertex
submission path instead of training on the local machine.
"""

from __future__ import annotations

from submit_vertex_retraining import main

if __name__ == "__main__":
    main()
