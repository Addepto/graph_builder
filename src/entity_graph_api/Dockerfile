########
# BASE #
########
FROM python:3.12-slim

WORKDIR /app/

# Set environment variables - fixed Poetry configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.5.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
COPY pyproject.toml ./
# Copy poetry.lock if it exists
COPY poetry.lock* ./

# Install dependencies globally since we're not using virtualenvs
RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --no-root --no-ansi --no-interaction

ARG commit
ARG version
ARG build_time

# Create build info
RUN echo "COMMIT: ${commit:-unknown}" > build_info && \
    echo "VERSION: ${version:-dev}" >> build_info && \
    echo "BUILD_TIME: ${build_time:-$(date)}" >> build_info

ENV COMMIT=${commit:-unknown}
ENV VERSION=${version:-dev}
ENV BUILD_TIME=${build_time:-$(date)}

# Copy application code
COPY . .

# Install the local project package into the image so top-level packages
# like `entity_graph` are available at runtime regardless of PYTHONPATH
# set by docker-compose. We run a full poetry install here (no-root omitted
# earlier to cache deps) which will install the package into the image.
RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --no-ansi --no-interaction