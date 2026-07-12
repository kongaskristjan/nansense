# The hosted nansense playground: a locked, shared MNIST + LeNet demo.
#
# Lives at the repository root because Hugging Face Docker Spaces only look
# for a root-level Dockerfile (there is no config key for a custom path).
# See deploy/README.md for the hosting side (Hugging Face Spaces or
# a plain VM):
#
#     docker build -t nansense-playground .
#     docker run --rm -p 7860:7860 nansense-playground
#
# Local builds need BuildKit (the docker-buildx-plugin): Docker's legacy
# builder, especially under a rootless daemon, writes layers the non-root
# USER below cannot read and dies at the first RUN after the user switch.
#
# The image bakes everything a cold start needs: `--prepare` runs the MNIST
# training at build time and freezes the run's last train batch — snapshot
# plus all layer statistics — into a moment file. Serving just reloads that
# moment and parks the locked session, so a container boots in seconds and
# needs neither the dataset nor a training pass.

FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Hugging Face Spaces runs containers as UID 1000 with no root fallback, so
# everything writable (the app dir, uv/matplotlib caches) lives under a home
# that user owns. Harmless on other hosts.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    MPLCONFIGDIR=/home/user/.cache/matplotlib
WORKDIR /home/user/app

# Dependency layer first, so code-only changes don't re-resolve torch.
COPY --chown=user:user pyproject.toml uv.lock ./
RUN uv sync --frozen --group cpu --no-install-project

COPY --chown=user:user . .
RUN uv sync --frozen --group cpu

# Train and freeze the demo moment. Serving reloads it without touching the
# dataset (see examples/playground/main.py).
RUN uv run --no-sync examples/playground/main.py --prepare --device cpu

EXPOSE 7860
CMD ["uv", "run", "--no-sync", "examples/playground/main.py", \
     "--device", "cpu", "--host", "0.0.0.0", "--nansense-port", "7860"]
