# ═══════════════════════════════════════════════════════════════
# NSMoR — Hermetic Reproducibility Container
# ═══════════════════════════════════════════════════════════════
#
# Build:  docker build -t nsmor .
# Test:   docker run --rm nsmor test
# Pipeline (GPU): docker compose run --rm nsmor pipeline
# Shell:  docker compose run --rm nsmor bash
# ═══════════════════════════════════════════════════════════════

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

LABEL maintainer="NSMoR Team"
LABEL description="Hermetic container for Tier-1 scientific reproducibility"

# ── System dependencies ─────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (layer-cached) ─────────────────────
WORKDIR /workspace

# Copy ONLY dependency metadata first.
# This layer is cached until pyproject.toml or requirements.txt change.
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


# ── Source code ─────────────────────────────────────────────
COPY . .

# Editable install (source now present) — reuses cached deps above.
RUN pip install --no-cache-dir -e ".[dev]"
RUN python -c "import torch, numpy as np; from torch.serialization import get_safe_globals, get_unsafe_globals_in_checkpoint, safe_globals, clear_safe_globals, add_safe_globals; assert torch.__version__.split('+')[0] == '2.6.0'; assert tuple(map(int, np.__version__.split('.')[:2])) >= (1, 26); print(torch.__version__, np.__version__)"

# ── Entrypoint ──────────────────────────────────────────────
ENTRYPOINT ["make"]
CMD ["help"]
