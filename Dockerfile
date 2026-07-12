# The hosted nansense playgrounds: locked, shared demos served from a frozen
# moment file (see examples/playground/main.py).
#
# Lives at the repository root because Hugging Face Docker Spaces only look
# for a root-level Dockerfile (there is no config key for a custom path).
# One image definition serves both Spaces: PLAYGROUND picks the demo, and
# deploy/push_space.sh stamps its default per Space. Locally:
#
#     docker build --build-arg PLAYGROUND=mnist -t nansense-playground .
#     docker run --rm -p 7860:7860 nansense-playground
#
# Local builds need BuildKit (the docker-buildx-plugin): Docker's legacy
# builder, especially under a rootless daemon, writes layers the non-root
# USER below cannot read and dies at the first RUN after the user switch.
#
# No training happens here: the moment file is trained locally on a GPU
# (`--prepare`, see deploy/README.md) and committed to the Space as a git-LFS
# object, so it arrives in the build context ready to serve. The container
# boots in seconds and needs neither the dataset nor a training pass.

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

ARG PLAYGROUND=imagenette
ENV PLAYGROUND=${PLAYGROUND}

# Fail the build early if the frozen moment is absent from the context, or is
# still a git-LFS pointer (a checkout without the LFS smudge filter).
RUN moment=".nansense_cache/playground/${PLAYGROUND}/moment.pt" && \
    test -f "$moment" || { echo "missing $moment — train it first, see deploy/README.md" >&2; exit 1; } && \
    ! head -c 12 "$moment" | grep -q "^version http" || { echo "$moment is an LFS pointer, not the moment — checkout needs git-lfs" >&2; exit 1; }

EXPOSE 7860
CMD exec uv run --no-sync examples/playground/main.py \
    --playground "$PLAYGROUND" --device cpu --host 0.0.0.0 --nansense-port 7860
