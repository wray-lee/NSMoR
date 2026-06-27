# ═══════════════════════════════════════════════════════════════
# NSMoR — Hermetic Reproducibility Container
# ═══════════════════════════════════════════════════════════════
#
# Build:  docker build -t nsmor .
# Test:   docker run --rm nsmor test
# Pipeline (GPU): docker compose run --rm nsmor pipeline
# Shell:  docker compose run --rm nsmor bash
# ═══════════════════════════════════════════════════════════════

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

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

# ── Entrypoint ──────────────────────────────────────────────
ENTRYPOINT ["make"]
CMD ["help"]
