import os

# Apply CUDA mask immediately when the package is imported.
_cuda_env = os.getenv("DEEPREADER_CUDA_VISIBLE_DEVICES")
if _cuda_env:
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_env
