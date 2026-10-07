"""Run the local model API with uvicorn."""

import uvicorn

from mena_mlops.serving.app import load_model_on_startup

if __name__ == "__main__":
    load_model_on_startup()
    uvicorn.run("mena_mlops.serving.app:app", host="0.0.0.0", port=8001)
