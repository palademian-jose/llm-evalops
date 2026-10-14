FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy LLM_PROVIDER=fake

COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --all-extras

COPY examples ./examples
COPY scripts ./scripts
COPY evals.yaml alerts.yaml ./

EXPOSE 8501
# Default: serve the dashboard on synthetic traces. Override with any `evalkit`/`llmops` command.
CMD ["sh", "-c", "uv run python scripts/synth_traces.py --out traces/synthetic && uv run llmops dashboard --traces traces/synthetic/baseline --compare traces/synthetic --port 8501"]
