# Mutistage build inspired by https://github.com/orgs/python-poetry/discussions/1879#discussioncomment-216865
# `python-base` sets up all our shared environment variables
FROM python:3.12-slim AS python-base

# python
ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # poetry
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=2.0.1 \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # make poetry create the virtual environment in the project's root
    # it gets named `.venv`
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    \
    # paths
    # this is where our requirements + virtual environment will live
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

# Prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# `builder-base` stage is used to build deps + create our virtual environment
FROM python-base AS builder-base
# Install dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    # deps for installing poetry
    curl \
    # deps for building python deps
    # gfortran libopenblas-dev cmake \
    gfortran libopenblas-dev coinor-libipopt-dev build-essential && \
    # dev deps
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python
# copy project requirement files here to ensure they will be cached.
WORKDIR $PYSETUP_PATH
# Copy project files
# TODO: return poetry.lock when versions are stable
COPY pyproject.toml README.md ./
COPY sippy/ sippy/

# install runtime deps - uses $POETRY_VIRTUALENVS_IN_PROJECT internally
RUN poetry install --without dev

# `development` image is used during development / testing
FROM python-base AS development
WORKDIR $PYSETUP_PATH

# Install git and pipx required for dev-container in VSCode
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    git pipx libopenblas-dev coinor-libipopt-dev

# copy in our built poetry + venv; copy in the source code
COPY --from=builder-base $POETRY_HOME $POETRY_HOME
COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH

# quicker install as runtime deps are already installed
RUN poetry install

# will become mountpoint of our code
WORKDIR /app
COPY . /app/

# # `production` image used for runtime
# FROM python-base AS production
# COPY --from=builder-base $PYSETUP_PATH $PYSETUP_PATH
# COPY . /app/
# WORKDIR /app

# Set the entrypoint
CMD ["poetry", "run", "python"]
