# =============================================================================
# sigdiscovPy + sigdiscov R Package Unified Docker Image
#
# Single Dockerfile for both CPU and GPU versions
#
# Build CPU version (default, Python only):
#   docker build -t sigdiscovpy:latest .
#
# Build GPU version:
#   docker build -t sigdiscovpy:gpu --build-arg USE_GPU=true .
#
# Build with R sigdiscov package (slower):
#   docker build -t sigdiscovpy:with-r --build-arg INSTALL_R=true .
#   docker build -t sigdiscovpy:gpu-with-r --build-arg USE_GPU=true --build-arg INSTALL_R=true .
#
# Run CPU:
#   docker run -it --rm -v $(pwd):/workspace sigdiscovpy:latest
#
# Run GPU:
#   docker run -it --rm --gpus all -v $(pwd):/workspace sigdiscovpy:gpu
#
# Run Jupyter:
#   docker run -it --rm -p 8888:8888 -v $(pwd):/workspace sigdiscovpy:latest \
#       jupyter lab --ip=0.0.0.0 --no-browser --allow-root
# =============================================================================

# Build arguments
ARG USE_GPU=false
ARG INSTALL_R=false

# =============================================================================
# Base Image Selection
# =============================================================================
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS base-true
FROM ubuntu:22.04 AS base-false

# Select base image based on USE_GPU argument
FROM base-${USE_GPU} AS base

# Re-declare ARGs after FROM (required by Docker)
ARG USE_GPU=false
ARG INSTALL_R=false

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# =============================================================================
# System Dependencies
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    # Build tools (required for Rcpp, RcppArmadillo, and C++ extensions)
    build-essential \
    gfortran \
    cmake \
    pkg-config \
    # Libraries for R packages and sigdiscov (Rcpp/RcppArmadillo)
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libhdf5-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libcairo2-dev \
    libxt-dev \
    # For R Matrix/linear algebra (required by sigdiscov)
    liblapack-dev \
    libblas-dev \
    # Utilities
    git \
    wget \
    curl \
    vim \
    locales \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Set locale (needed for some R packages)
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# =============================================================================
# R: Install R and Packages (optional)
# =============================================================================

ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing R from Ubuntu repos..." && \
        echo "========================================" && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            r-base \
            r-base-dev && \
        rm -rf /var/lib/apt/lists/* && \
        R -e "cat('R version:', R.version.string, '\n')"; \
    else \
        echo "Skipping R installation"; \
    fi

# Use Posit Package Manager for pre-compiled R binaries (much faster than source)
ENV RSPM="https://packagemanager.posit.co/cran/__linux__/jammy/latest"

# Install CRAN dependencies for sigdiscov (pre-compiled binaries)
# sigdiscov Imports: Matrix, Rcpp, RcppArmadillo, methods, stats, utils
# sigdiscov Suggests: Seurat, rhdf5, ggplot2, sctransform, testthat
# sigdiscov LinkingTo: Rcpp, RcppArmadillo
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing CRAN packages (binary)..." && \
        echo "========================================" && \
        R -e "options(repos = c(CRAN = Sys.getenv('RSPM', 'https://cloud.r-project.org/'))); \
              install.packages(c( \
                  'remotes', 'BiocManager', 'devtools', \
                  'Matrix', 'Rcpp', 'RcppArmadillo', \
                  'ggplot2', 'sctransform', \
                  'testthat', 'data.table', \
                  'Seurat', 'hdf5r' \
              ), dependencies = TRUE, Ncpus = parallel::detectCores())"; \
    fi

# Install Bioconductor packages (rhdf5 for HDF5 I/O cross-validation)
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing Bioconductor packages..." && \
        echo "========================================" && \
        R -e "BiocManager::install(ask = FALSE, update = FALSE)" && \
        R -e "BiocManager::install(c( \
                  'rhdf5' \
              ), ask = FALSE, update = FALSE, Ncpus = parallel::detectCores())"; \
    fi

# Install sigdiscov from GitHub (psychemistz/sigdiscov)
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Installing sigdiscov from GitHub..." && \
        echo "========================================" && \
        R -e "options(timeout = 600); \
              remotes::install_github('psychemistz/sigdiscov', \
                  dependencies = TRUE, \
                  upgrade = 'never', \
                  force = TRUE); \
              library(sigdiscov); \
              cat('sigdiscov version:', as.character(packageVersion('sigdiscov')), '\n')"; \
    fi

# Verify R installation (fail build if required packages are missing)
ARG INSTALL_R
RUN if [ "$INSTALL_R" = "true" ]; then \
        echo "========================================" && \
        echo "Verifying R installation..." && \
        echo "========================================" && \
        R -e "cat('R version:', R.version.string, '\n'); \
              required <- c('sigdiscov', 'Matrix', 'Rcpp', 'RcppArmadillo'); \
              all_ok <- TRUE; \
              for (pkg in required) { \
                  if (requireNamespace(pkg, quietly = TRUE)) { \
                      cat(pkg, as.character(packageVersion(pkg)), 'OK\n') \
                  } else { \
                      cat(pkg, 'MISSING\n'); \
                      all_ok <- FALSE \
                  } \
              }; \
              if (!all_ok) stop('Required R packages are missing!')"; \
    fi

# =============================================================================
# Python: Install sigdiscovPy
# =============================================================================

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install base Python packages
RUN pip3 install --no-cache-dir \
    "numpy>=1.21.0" \
    "scipy>=1.7.0" \
    "pandas>=1.3.0" \
    "anndata>=0.8.0" \
    "h5py>=3.0.0" \
    "tqdm>=4.60.0" \
    "scikit-learn>=1.0.0" \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab

# Install CuPy for GPU version only
ARG USE_GPU
RUN if [ "$USE_GPU" = "true" ]; then \
        echo "Installing CuPy for GPU support..." && \
        pip3 install --no-cache-dir cupy-cuda12x; \
    else \
        echo "Skipping CuPy (CPU-only build)"; \
    fi

# Install sigdiscovPy from official GitHub repository (always latest version)
RUN pip3 install --no-cache-dir git+https://github.com/psychemistz/sigdiscovPy.git

# Verify Python installation
RUN python3 -c "import sigdiscovpy; print(f'sigdiscovPy {sigdiscovpy.__version__} OK, GPU: {sigdiscovpy.GPU_AVAILABLE}')"

# =============================================================================
# Environment
# =============================================================================

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# GPU environment variables (harmless if not using GPU)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# =============================================================================
# Entry Point
# =============================================================================

CMD ["/bin/bash"]

# =============================================================================
# Labels
# =============================================================================

LABEL maintainer="Seongyong Park <https://github.com/psychemistz>"
LABEL description="sigdiscovPy - Spatial Signature Discovery (CPU/GPU)"
LABEL version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/psychemistz/sigdiscovPy"
