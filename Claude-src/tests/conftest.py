import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# config.py auto-detects Kaggle (/kaggle/input) vs. a hardcoded WSL path
# (/mnt/c/Users/.../Claude-src) and creates cache/output/models dirs under
# whichever it picks -- neither exists in an arbitrary CI/sandbox environment.
# Any test file that imports one of the numbered scripts (e.g.
# test_llm_batching.py importing 02_llm_enrichment) transitively imports
# config.py, so give it a writable scratch dir here, ONCE, unless the caller
# already set these (e.g. a real Kaggle run never needs this, IS_KAGGLE wins).
os.environ.setdefault("TY_WORK_DIR", tempfile.mkdtemp(prefix="ty_test_work_"))
os.environ.setdefault("TY_DATA_DIR", tempfile.mkdtemp(prefix="ty_test_data_"))
