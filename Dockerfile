# Use the Python 3.12 image from the Docker Hub
FROM python:3.12 AS base

# Label the image
LABEL maintainer="James J"
LABEL version="1.0"
LABEL description="This is a devcontainer image for python3 development for Control System"

# Set non-interactive frontend
ARG DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt-get update && \
    apt-get install -y \
    curl \
    git \
    sudo \
    build-essential \
    cmake \
    ninja-build \
    libblas-dev \
    liblapack-dev \
    gcc \
    gfortran \
    gdb \
    tree \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# Install UV for modern Python dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /home/vscode/.cargo/bin/uv tool install --global jupyterlab && \
    export PATH="/home/vscode/.cargo/bin:$PATH" && \
    /home/vscode/.cargo/bin/uv pip install --system slycot control harold

# Create a non root user vscode, set zsh as the default shell, and add to sudo group
RUN useradd -m -s /bin/zsh vscode && echo "vscode ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Switch to the vscode user
USER vscode

# Install Oh My Zsh and set the theme to "arrow"
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    sed -i 's/ZSH_THEME=".*"/ZSH_THEME="arrow"/' /home/vscode/.zshrc

# Set the working directory
WORKDIR /workspace

# Expose JupyterLab port
EXPOSE 8888
