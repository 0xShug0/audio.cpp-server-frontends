# syntax=docker/dockerfile:1
# Context contains sibling audio.cpp/ and frontends/ checkouts.
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:13.0.2-runtime-ubuntu24.04@sha256:6a0e31b59e70890446f2c17356d6efc0d54260090a8c63d6ca7c2ad049db95d2
FROM ${BASE_CUDA_DEV_CONTAINER} AS build

ARG GCC_VERSION=14

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc-${GCC_VERSION} g++-${GCC_VERSION} cmake ninja-build git ca-certificates libmp3lame-dev \
    && rm -rf /var/lib/apt/lists/*
ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} CUDAHOSTCXX=g++-${GCC_VERSION}

COPY audio.cpp/ /src/audio.cpp/
COPY frontends/modules/ /src/frontends/modules/
COPY frontends/external/ /src/frontends/external/
COPY frontends/server_frontends.cmake /src/frontends/server_frontends.cmake
WORKDIR /src/audio.cpp

# Use audio.cpp's portable CUDA architecture list, including DGX Spark.
# Limit parallelism to avoid exhausting hosted runner RAM during CUDA builds.
RUN cmake -S . -B build/debug -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DAUDIOCPP_DEPLOYMENT_BUILD=ON \
    -DAUDIOCPP_MODEL_SET=full \
    -DENGINE_ENABLE_NATIVE_CPU=OFF \
    -DENGINE_ENABLE_CPU_ALL_VARIANTS=ON \
    -DENGINE_ENABLE_CUDA=ON \
    -DENGINE_ENABLE_CUDA_GRAPHS=ON \
    -DENGINE_ENABLE_VULKAN=OFF \
    -DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON \
    -DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON \
    -DAUDIOCPP_SERVER_FRONTENDS_DIR=/src/frontends \
    '-DAUDIOCPP_SERVER_FRONTEND_MODULES=audio_decode;mp3_encode;https;websocket' \
    -DENGINE_BUILD_EXAMPLES=OFF \
    -DENGINE_BUILD_TESTS=OFF \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build/debug --parallel 2 --target audiocpp_server audiocpp_cli

RUN mkdir -p /out && \
    cp build/debug/bin/audiocpp_server build/debug/bin/audiocpp_cli /out/ && \
    find build/debug -name '*.so*' -exec cp -P {} /out/ \;

FROM ${BASE_CUDA_RUN_CONTAINER}

ARG AUDIOCPP_REVISION
ARG FRONTENDS_REVISION
ARG IMAGE_SOURCE
LABEL org.opencontainers.image.source=$IMAGE_SOURCE \
      org.opencontainers.image.revision=$FRONTENDS_REVISION \
      org.opencontainers.image.title="audio.cpp with all server frontends" \
      io.audiocpp.core.revision=$AUDIOCPP_REVISION

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libmp3lame0 libespeak-ng1 ca-certificates curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /app/
COPY audio.cpp/model_specs/ /app/model_specs/
COPY audio.cpp/assets/ /app/assets/
COPY audio.cpp/LICENSE /usr/share/doc/audio.cpp/LICENSE
COPY frontends/LICENSE /usr/share/doc/audio.cpp-server-frontends/LICENSE
ENV LD_LIBRARY_PATH=/app:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
WORKDIR /app

# NVIDIA's driver library is supplied by the host at container launch.
RUN ldd /app/audiocpp_server > /app/runtime-dependencies.txt && \
    cat /app/runtime-dependencies.txt && \
    ! grep 'not found' /app/runtime-dependencies.txt | grep -v 'libcuda.so.1'

RUN mkdir -p /app/models && chown ubuntu:ubuntu /app/models
USER ubuntu
EXPOSE 8080
ENTRYPOINT ["/app/audiocpp_server"]
CMD ["--ui", "--ui-management", "--host", "0.0.0.0", "--port", "8080", "--backend", "cuda", "--log"]
