
## How to initialize your environment to run gRPC code
1. Install miniconda ([link](https://www.anaconda.com/docs/getting-started/miniconda/install))
2. Run the following commands to create the `grpc-py39` environment.
```bash
cd panoseti/grpc
conda create -n grpc-py39 python=3.9
conda activate grpc-py39
conda install -c conda-forge grpcio-tools  # get google/proto files
pip install -r requirements
```