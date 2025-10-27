#!/bin/bash

# Wow, squeezing concurrency out of llama-server is hard!
# TODO: Download a model to use with the python-llama-cpp project and try to it's server?

. ~/profile.d/cuda

PROJECT_DIR="$(pwd)"

# sudo apt-get install libcurl4-openssl-dev -y

die() {
    echo "$1"
    exit 1
}

# brew install llama.cpp
# .. is the easy way out but doesn't provide CUDA support. Likely because CUDA isn't freely distributable?
# So we build from source for CUDA support
# brew uninstall llama.cpp

cd /tmp/
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
fi
cd llama.cpp

# Set the environment variable for the build
export GGML_CUDA=1

# 1. Create a build directory and navigate into it
mkdir build
cd build

if [ -f "bin/llama-cli" ]; then
    echo "llama-cli already built"
else
    # 2. Run cmake to configure the project, enabling CUDA
    if [ ! -f "MakeFile" ]; then
        cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=all-major

        if [ ! -f "MakeFile" ]; then
            die "Failed to configure cmake"
        fi
    fi

    # Compile the project using multiple threads for speed
    cmake --build . --config Release -j24 | tee build.out
fi

# Simple test to see if it works
if [ ! -f "bin/llama-cli" ]; then
    die "Failed to build llama-cli"
fi
echo "Are you alive?" | ./bin/llama-cli -m $PROJECT_DIR/../models/unsloth_Qwen3-1.7B-GGUF_Qwen3-1.7B-Q4_K_M.gguf -ngl 999 

if [ $? -ne 0 ]; then
    echo "Chat completion with $(pwd)/bin/llama-cli failed"
    die "Review the build and try again?"
fi

if [ ! -f "/usr/local/bin/llama-server" ]; then
    sudo make install
fi
