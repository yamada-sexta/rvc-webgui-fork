# RVC WebGUI Fork

<div align="center">

[![License](https://img.shields.io/github/license/yamada-sexta/rvc-webgui-fork?style=flat-square)](https://github.com/yamada-sexta/rvc-webgui-fork/blob/main/THIRD_PARTY_NOTICES.md)
[![GitHub stars](https://img.shields.io/github/stars/yamada-sexta/rvc-webgui-fork?style=flat-square)](https://github.com/yamada-sexta/rvc-webgui-fork/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/yamada-sexta/rvc-webgui-fork?style=flat-square)](https://github.com/yamada-sexta/rvc-webgui-fork/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/yamada-sexta/rvc-webgui-fork?style=flat-square)](https://github.com/yamada-sexta/rvc-webgui-fork/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/yamada-sexta/rvc-webgui-fork?style=flat-square)](https://github.com/yamada-sexta/rvc-webgui-fork/pulls)

</div>

This fork aims to improve upon the [original RVC project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) and with several enhancements:

- Better dependency management

- Improved Gradio UI

- Better Docker support

## Getting Started

### Run on Google Colab

Follow instructions on this [notebook](https://colab.research.google.com/github/yamada-sexta/rvc-webgui-fork/blob/main/notebook/colab.ipynb).

### Run locally

> Install [uv](https://docs.astral.sh/uv/#installation) if you haven't already.

```bash
git clone https://github.com/yamada-sexta/rvc-webgui-fork.git
cd rvc-webgui-fork
uv sync --locked
uv run ./tools/download_models.py
uv run ./web_ui.py
```

### Docker Compose

Create a `docker-compose.yml` similar to this:

```yml
services:
  rvc-fork-server:
    image: ghcr.io/yamada-sexta/rvc-webgui-fork:latest
    # Or if you want the absolute latest version
    # build:
    #   context: https://github.com/yamada-sexta/rvc-webgui-fork.git
    #   dockerfile: Dockerfile
    restart: "unless-stopped"
    shm_size: '16gb'
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "7865:7865"
    volumes:
      - ./data/cache:/cache
      - ./data/datasets:/app/datasets
      - ./data/weights:/app/assets/weights
      - ./data/logs:/app/logs
      - /app/logs/mute
```

In the same directory run

```bash
docker-compose up -d
```

Then visit `localhost:7865` in your browser.

## Contributing

We have a more open policy to contribution compared to the original.

Please refer [this guide](./CONTRIBUTING.md) for more details.
