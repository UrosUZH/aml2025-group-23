# SignCLIP Docker Images

[SignCLIP](https://github.com/J22Melody/fairseq/tree/main/examples/MMPT) is a project designed for evaluating various models on sign language datasets. However, the original repository did not include Docker support, and the inference and training environments required different dependencies.

To simplify testing and development, we created Docker images for both environments, making it easy to run SignCLIP on any machine with Docker support.

## Build Docker Images Locally

You can build the Docker images directly from the provided Dockerfiles:

### Training Environment

```bash
docker build -f Dockerfile.dev -t signclip_dev:1.0 .
```

### Inference Environment

```bash
docker build -f Dockerfile.inference -t signclip_inf:1.0 .
```

---

## Pull Prebuilt Docker Images

If you prefer not to build the images yourself, you can pull the prebuilt versions from Docker Hub:

```bash
docker pull rohamzn/signclip_dev:1.0
docker pull rohamzn/signclip_inf:1.0
```

---

## Run the Docker Images

To run the containers (with GPU support), use the following commands:

```bash
docker run --gpus all -it --name signclip_dev rohamzn/signclip_dev:1.0 bash
docker run --gpus all -it --name signclip_inf rohamzn/signclip_inf:1.0 bash
```

---

Let me know if you'd like to include volume mounting, port forwarding, or `docker-compose` support.
