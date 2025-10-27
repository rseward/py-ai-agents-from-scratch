# Purpose

A python version of this node project:
- https://github.com/pguso/ai-agents-from-scratch

This project is a simple translation of the code to use python-llama-cpp instead of node-llama-cpp.


```

export CC=gcc-12
export CXX=g++-12

uv pip install llama-cpp-python[server] --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125
#uv pip install llama-cpp-python==0.3.13 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125
```