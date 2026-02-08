# Docker Usage Guide

This guide explains how to use sigdiscovPy with Docker, including CPU, GPU, and R-enabled versions.

## Quick Start

### Pull Pre-built Images

```bash
# CPU version (Python only) - recommended for most users
docker pull psychemistz/sigdiscovpy:latest

# GPU version (Python + CuPy)
docker pull psychemistz/sigdiscovpy:gpu

# CPU + R version (Python + sigdiscov R package)
docker pull psychemistz/sigdiscovpy:with-r

# GPU + R version (full stack)
docker pull psychemistz/sigdiscovpy:gpu-with-r
```

### Build Images Locally

```bash
# CPU version (default, Python only ~5 min)
docker build -t sigdiscovpy:latest .

# GPU version (Python + CuPy ~10 min)
docker build -t sigdiscovpy:gpu --build-arg USE_GPU=true .

# CPU + R version (Python + R packages ~30-60 min)
docker build -t sigdiscovpy:with-r --build-arg INSTALL_R=true .

# GPU + R version (full stack ~40-90 min)
docker build -t sigdiscovpy:gpu-with-r --build-arg USE_GPU=true --build-arg INSTALL_R=true .
```

### Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `USE_GPU` | `false` | Set to `true` for GPU/CUDA support |
| `INSTALL_R` | `false` | Set to `true` to include R and sigdiscov package |

## Available Docker Images

| Tag | Python | GPU | R | Size | Use Case |
|-----|--------|-----|---|------|----------|
| `latest` / `cpu` | yes | no | no | ~2 GB | General use |
| `gpu` | yes | yes | no | ~6 GB | GPU acceleration |
| `with-r` / `cpu-with-r` | yes | no | yes | ~3.5 GB | R cross-validation |
| `gpu-with-r` | yes | yes | yes | ~8 GB | Full stack |

## R Packages Included

When building with `INSTALL_R=true`, the following packages are installed. CRAN packages are fetched as pre-compiled binaries from the [Posit Public Package Manager](https://packagemanager.posit.co/) for faster builds.

### From CRAN (pre-compiled binaries)

**Core (sigdiscov dependencies):**
- Matrix, Rcpp, RcppArmadillo (required by sigdiscov)
- ggplot2, sctransform (suggested by sigdiscov)
- testthat, data.table
- remotes, BiocManager, devtools

**Optional:**
- Seurat, hdf5r (for Seurat/HDF5 workflows)

### From Bioconductor (`BiocManager::install()`)
- rhdf5 (HDF5 I/O for cross-validation with Python)

### From GitHub (`remotes::install_github()`)
- **sigdiscov**: `psychemistz/sigdiscov` - Spatial signature discovery R package

## Running Containers

### Interactive Mode

```bash
# CPU
docker run -it --rm -v $(pwd):/workspace psychemistz/sigdiscovpy:latest

# GPU
docker run -it --rm --gpus all -v $(pwd):/workspace psychemistz/sigdiscovpy:gpu

# CPU + R (for cross-validation)
docker run -it --rm -v $(pwd):/workspace psychemistz/sigdiscovpy:with-r
```

### Run Scripts

```bash
# Python script
docker run --rm -v $(pwd):/workspace psychemistz/sigdiscovpy:latest python your_script.py

# R script
docker run --rm -v $(pwd):/workspace psychemistz/sigdiscovpy:with-r Rscript your_script.R
```

## Using Docker Compose

Docker Compose simplifies container management:

```bash
# Start CPU container
docker-compose up -d sigdiscovpy

# Start GPU container
docker-compose up -d sigdiscovpy-gpu

# Start CPU + R container
docker-compose up -d sigdiscovpy-r

# Enter a running container
docker-compose exec sigdiscovpy bash

# Stop all containers
docker-compose down
```

### Available Services

| Service | GPU | R | Port | Description |
|---------|-----|---|------|-------------|
| `sigdiscovpy` | no | no | - | CPU interactive |
| `sigdiscovpy-gpu` | yes | no | - | GPU interactive |
| `sigdiscovpy-r` | no | yes | - | CPU + R interactive |
| `sigdiscovpy-gpu-r` | yes | yes | - | GPU + R interactive |
| `sigdiscovpy-jupyter` | no | no | 8888 | Jupyter Lab (CPU) |
| `sigdiscovpy-jupyter-gpu` | yes | no | 8889 | Jupyter Lab (GPU) |
| `sigdiscovpy-jupyter-r` | no | yes | 8890 | Jupyter Lab (CPU + R) |

### Jupyter Lab

```bash
# Start Jupyter (CPU)
docker-compose up sigdiscovpy-jupyter
# Open http://localhost:8888

# Start Jupyter (GPU)
docker-compose up sigdiscovpy-jupyter-gpu
# Open http://localhost:8889

# Start Jupyter (CPU + R)
docker-compose up sigdiscovpy-jupyter-r
# Open http://localhost:8890
```

## Validation Examples

### Verify Python Installation

```bash
docker run --rm psychemistz/sigdiscovpy:latest python -c "
import sigdiscovpy
print(f'sigdiscovPy {sigdiscovpy.__version__}')
print(f'GPU available: {sigdiscovpy.GPU_AVAILABLE}')
"
```

### Verify R Installation

```bash
docker run --rm psychemistz/sigdiscovpy:with-r Rscript -e "
cat('R version:', R.version.string, '\n')

# Check packages
for (pkg in c('sigdiscov', 'Matrix', 'Rcpp', 'RcppArmadillo')) {
    if (require(pkg, quietly = TRUE, character.only = TRUE)) {
        cat(pkg, 'OK -', as.character(packageVersion(pkg)), '\n')
    } else {
        cat(pkg, 'NOT FOUND\n')
    }
}
"
```

### Cross-Validation (R vs Python)

```bash
# Enter container with R
docker run -it --rm -v $(pwd):/workspace psychemistz/sigdiscovpy:with-r

# Inside container - Run R computation
Rscript -e "
library(sigdiscov)
# ... your R analysis code
"

# Run Python computation
python -c "
import sigdiscovpy
# ... your Python analysis code
"
```

## GPU Support

### Prerequisites

1. NVIDIA GPU with CUDA support
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Installation (Ubuntu)

```bash
# Add NVIDIA repository
distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### Verify GPU Access

```bash
# Check GPU is accessible
docker run --rm --gpus all psychemistz/sigdiscovpy:gpu nvidia-smi

# Check CuPy
docker run --rm --gpus all psychemistz/sigdiscovpy:gpu python -c "
import cupy as cp
print(f'CuPy version: {cp.__version__}')
x = cp.arange(10)
print(f'GPU array: {x}')
"
```

## Troubleshooting

### R Package Installation Failures

If R packages fail to install, try building with verbose output:

```bash
docker build -t sigdiscovpy:with-r --build-arg INSTALL_R=true --progress=plain .
```

Common issues:
- **Network timeout**: Increase timeout in Dockerfile or retry
- **Missing system library**: Add to apt-get install list
- **GitHub rate limit**: Wait and retry, or use a GitHub token

### Permission Issues

```bash
# Run as current user
docker run -it --rm -u $(id -u):$(id -g) -v $(pwd):/workspace psychemistz/sigdiscovpy:latest
```

### Memory Issues

```bash
# Increase memory limit
docker run -it --rm -m 16g -v $(pwd):/workspace psychemistz/sigdiscovpy:latest
```

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker info | grep -i gpu

# Verify nvidia-container-toolkit
which nvidia-container-toolkit
```

## Customization

### Add Your Own Packages

Create a custom Dockerfile:

```dockerfile
FROM psychemistz/sigdiscovpy:with-r

# Add R packages
RUN R -e "install.packages('your_r_package', repos='https://cloud.r-project.org/')"

# Add Python packages
RUN pip3 install your_python_package
```

### Mount Additional Data

```bash
docker run -it --rm \
  -v $(pwd):/workspace \
  -v /path/to/data:/data:ro \
  -v /path/to/results:/results \
  psychemistz/sigdiscovpy:latest
```

## Singularity / HPC

On HPC clusters where Docker is not available, use Singularity or Apptainer to pull the Docker images:

```bash
# Load Singularity or Apptainer (cluster-specific)
module load singularity   # or: module load apptainer

# Pull CPU image
singularity pull docker://psychemistz/sigdiscovpy:latest

# Pull R-enabled image
singularity pull docker://psychemistz/sigdiscovpy:with-r

# Run interactively
singularity exec sigdiscovpy_latest.sif python3 -c "import sigdiscovpy; print(sigdiscovpy.__version__)"

# Run with data binding
singularity exec --bind /path/to/data:/data sigdiscovpy_latest.sif python3 your_script.py

# Run R inside the container
singularity exec sigdiscovpy_with-r.sif Rscript -e "library(sigdiscov); cat('OK\n')"
```

> **Note:** Building images from the Dockerfile on HPC requires `fakeroot` support. If unavailable, pull the pre-built images from Docker Hub instead.

## CI/CD Notes

- Default CI builds use `INSTALL_R=false` for speed
- R-enabled builds are triggered on releases or manual dispatch
- Use `workflow_dispatch` with `build_r=true` to build R images manually

```yaml
# Trigger R build manually in GitHub Actions
workflow_dispatch:
  inputs:
    build_r:
      description: 'Build images with R packages'
      default: 'true'
```
