# Barba

Barba is a multilingual [natural language inference](http://nlpprogress.com/english/natural_language_inference.html) model for [textual entailment](https://en.wikipedia.org/wiki/Textual_entailment) and [zero-shot text classification](https://joeddav.github.io/blog/2020/05/29/ZSL.html#Classification-as-Natural-Language-Inference), available as an end-to-end service through TensorFlow Serving. Based on [XLM-RoBERTa](https://arxiv.org/abs/1911.02116), it is trained on selected subsets of publicly available English ([GLUE](https://huggingface.co/datasets/glue)), Chinese ([CLUE](https://huggingface.co/datasets/clue)), Japanese ([JGLUE](https://huggingface.co/datasets/shunk031/JGLUE)), Korean ([KLUE](https://huggingface.co/datasets/klue)) datasets, as well as other private datasets.

## Quick Start

### Installation

Docker images are available on [Docker Hub](https://hub.docker.com/r/hyperonym/barba/tags) and [GitHub Packages](https://github.com/orgs/hyperonym/packages?repo_name=barba).

For GPU acceleration, you also need to install the [NVIDIA Driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html) and [NVIDIA Container Runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Barba's image already comes with related libraries such as CUDA and cuDNN, so there is no need to install them manually.

### Basic Usage

Replace `X.Y.Z` with the [latest version](https://hub.docker.com/r/hyperonym/barba/tags), then run:

```bash
docker run -p 8501:8501 hyperonym/barba:X.Y.Z
```

#### Textual Entailment

```bash
curl -X POST \
  'http://127.0.0.1:8501/v1/models/barba:predict' \
  --data-raw '{
  "instances": [
    {
      "hypothesis": "Leonardo da Vinci never painted on walnut wood",
      "premise": "The Lady with an Ermine is a portrait painting widely attributed to the Italian Renaissance artist Leonardo da Vinci. Dated to around 1489 to 1491, the work is painted in oils on a panel of walnut wood."
    },
    {
      "hypothesis": "最长的黄鳝能有1米长",
      "premise": "黄鳝（学名：Monopterus albus）：又名鳝鱼，古代称为蝉鱼、黄蝉等。体细长呈蛇形，体长约20-70厘米，最长可达1米。体前圆后部侧扁，尾尖细。"
    }
  ]
}'
```

#### Zero-shot Text Classification

```bash
curl -X POST \
  'http://127.0.0.1:8501/v1/models/barba:predict' \
  --data-raw '{
  "signature_name": "cartesian",
  "inputs": {
    "hypothesis": [
      "foreign policy",
      "Europe",
      "elections",
      "business",
      "outdoor recreation",
      "politics"
    ],
    "premise": [
      "Who are you voting for in 2020?",
      "北欧有很多不错的滑雪场"
    ]
  }
}'
```

## License

Barba is available under the [MIT License](https://github.com/hyperonym/barba/blob/master/LICENSE).

---

© 2023 [Hyperonym](https://hyperonym.org)
