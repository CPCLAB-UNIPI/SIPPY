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
    python3-dev \
    python3-pytest \
    && rm -rf /var/lib/apt/lists/*

# Create a non root user vscode, set zsh as the default shell, and add to sudo group
RUN useradd -m -s /bin/zsh vscode && echo "vscode ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Install UV for modern Python dependency management
USER root
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mkdir -p /home/vscode/.local/bin /home/vscode/.local/share/uv && \
    cp /root/.local/bin/* /home/vscode/.local/bin/ && \
    chown -R vscode:vscode /home/vscode/.local

# Switch to the vscode user
USER vscode

# Add uv to PATH in .zshrc and install packages
RUN echo 'export PATH="/home/vscode/.local/bin:$PATH"' >> /home/vscode/.zshrc && \
    export PATH="/home/vscode/.local/bin:$PATH" && \
    /home/vscode/.local/bin/uv tool install jupyterlab
    
USER root
RUN /root/.local/bin/uv pip install --system slycot control harold
    
USER vscode

# Install Oh My Zsh and set the theme to "arrow"
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    sed -i 's/ZSH_THEME=".*"/ZSH_THEME="arrow"/' /home/vscode/.zshrc && \
    echo 'export PATH="/home/vscode/.local/bin:$PATH"' >> /home/vscode/.zshrc

# Set the working directory
WORKDIR /workspace

# Expose JupyterLab port
EXPOSE 8888
