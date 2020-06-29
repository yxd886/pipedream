#
# This example Dockerfile illustrates a method to apply
# patches to the source code in NVIDIA's PyTorch
# container image and to rebuild PyTorch.  The RUN command
# included below will rebuild PyTorch in the same way as
# it was built in the original image.
#
# By applying customizations through a Dockerfile and
# `docker build` in this manner rather than modifying the
# container interactively, it will be straightforward to
# apply the same changes to later versions of the PyTorch
# container image.
#
# https://docs.docker.com/engine/reference/builder/
#
FROM nvcr.io/nvidia/pytorch:19.09-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        texlive-latex-extra \
      && \
    rm -rf /var/lib/apt/lists/

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Reset default working directory
WORKDIR /workspace
