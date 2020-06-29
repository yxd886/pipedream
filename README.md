# PipeDream-2BW: Memory-Efficient Pipeline-Parallel Training for Large DNNs

This repository contains the source code implementation for the ArXiv paper
["Memory-Efficient Pipeline-Parallel Training for Large DNNs"](https://arxiv.org/abs/2006.09503).

## Directory Structure

### `planner`

A Python implementation of PipeDream-2BW's planner.

### `runtime`

PipeDream-2BW's runtime, which implements model parallelism, as well as input
pipelining in PyTorch. This can be fused with data parallelism to give hybrid
model and data parallelism, and input pipelining.

## Setup

### Software Dependencies

To run PipeDream-2BW, you will need a NVIDIA GPU with CUDA 10.0, GPU driver version 418.56, nvidia-docker2,
and Python 3. On a Linux server with NVIDIA GPU(s) and Ubuntu 16.04, these dependencies can be installed
using,

```bash
bash setup.sh
```

All dependencies are in the nvcr.io/nvidia/pytorch:19.09-py3 container, which can be downloaded using,

```bash
nvidia-docker pull nvcr.io/nvidia/pytorch:19.09-py3
```

The PyTorch Docker Container can then be run using,

```bash
nvidia-docker run -it -v /mnt:/mnt --ipc=host --net=host <CONTAINER_NAME> /bin/bash
```


## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.


## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](LICENSE.txt) license.

