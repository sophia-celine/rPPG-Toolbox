```bash
#!/bin/bash

# Check if a mode argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 {conda|uv}"
    exit 1
fi

MODE=$1

# Function to set up using conda
conda_setup() {
    echo "Setting up using conda..."

    # Remove existing environment
    conda env remove --name rppg-toolbox311 -y >/dev/null 2>&1 || true

    # Create environment
    conda create -n rppg-toolbox311 python=3.11 -y || exit 1

    # Initialize and activate conda
    source "$(conda info --base)/etc/profile.d/conda.sh" || exit 1
    conda activate rppg-toolbox311 || exit 1

    # Install CUDA 12.8 compiler
    conda install -c nvidia cuda-nvcc=12.8 -y || exit 1

    # Install GCC/G++ 13
    conda install -c conda-forge \
        gcc_linux-64=13 \
        gxx_linux-64=13 \
        -y || exit 1

    # Force the build to use CUDA 12.8 from this Conda environment
    export CUDA_HOME="$CONDA_PREFIX"

    # Remove Windows CUDA installations from PATH
    export PATH=$(echo "$PATH" | tr ':' '\n' | \
        grep -v '/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/' | \
        paste -sd:)

    # Put the Conda environment first in PATH
    export PATH="$CUDA_HOME/bin:$PATH"

    # Force the Conda GCC/G++ compilers
    export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
    export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"

    # Make CUDA and PyTorch libraries visible at runtime
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

    echo ""
    echo "===== CUDA / Compiler configuration ====="
    echo "CUDA_HOME=$CUDA_HOME"
    echo "nvcc=$(which nvcc)"
    nvcc --version || exit 1
    echo "CC=$CC"
    echo "CXX=$CXX"
    $CC --version || exit 1
    $CXX --version || exit 1
    echo "========================================="
    echo ""

    # Install PyTorch with CUDA 12.8
    pip install torch==2.11.0+cu128 \
        torchvision==0.26.0+cu128 \
        torchaudio==2.11.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 || exit 1

    # Install regular requirements, excluding packages that need CUDA compilation
    grep -Ev '^(causal-conv1d|mamba-ssm)==' requirements.txt | \
        pip install -r /dev/stdin || exit 1

    # Build CUDA extensions using the CUDA 12.8 environment
    pip install --no-build-isolation --no-cache-dir \
        causal-conv1d==1.7.0 \
        mamba-ssm==2.3.2.post1 || exit 1

    # Verify Python packages
    echo ""
    echo "===== Verifying Python packages ====="

#     python - <<'PY'
# import sys
# import torch
# import torchvision
# import torchaudio
# import numpy
# import scipy
# import cv2
# import mediapipe

# print("Python:", sys.version)
# print("PyTorch:", torch.__version__)
# print("TorchVision:", torchvision.__version__)
# print("TorchAudio:", torchaudio.__version__)
# print("Torch CUDA:", torch.version.cuda)
# print("CUDA available:", torch.cuda.is_available())

# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))

# print("NumPy:", numpy.__version__)
# print("SciPy:", scipy.__version__)
# print("OpenCV:", cv2.__version__)
# print("MediaPipe:", mediapipe.__version__)
# PY

    # Verify package dependencies
    pip check || exit 1

    # Verify CUDA compiler
    echo ""
    echo "Checking NVCC..."
    nvcc --version || exit 1

    # Verify causal-conv1d
    echo ""
    echo "Checking causal-conv1d..."
    python -c "import causal_conv1d; print('causal-conv1d OK')" || exit 1

    # Verify mamba-ssm
    echo ""
    echo "Checking mamba-ssm..."
    python -c "import mamba_ssm; print('mamba-ssm OK')" || exit 1

    echo ""
    echo "Installation completed successfully!"
}

# Execute the appropriate setup based on the mode
case $MODE in
    conda)
        conda_setup
        ;;
    uv)
        uv_setup
        ;;
    *)
        echo "Invalid mode: $MODE"
        echo "Usage: $0 {conda|uv}"
        exit 1
        ;;
esac
```
