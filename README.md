# Barba

Barba is a multilingual model for [Natural Language Inference](https://paperswithcode.com/task/natural-language-inference) (NLI) and [zero-shot text classification](https://joeddav.github.io/blog/2020/05/29/ZSL.html#Classification-as-Natural-Language-Inference), available as an end-to-end service through TensorFlow Serving. It is based on [XLM-RoBERTa](https://arxiv.org/abs/1911.02116), and is trained on selected subsets of [SNLI](https://nlp.stanford.edu/projects/snli/), [XNLI](https://github.com/facebookresearch/XNLI), [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/), [OCNLI](https://github.com/CLUEbenchmark/OCNLI), [ANLI](https://github.com/facebookresearch/anli), and other private datasets.

## Quick Start

### Installation

Docker images are available on [Docker Hub](https://hub.docker.com/r/hyperonym/barba/tags) and [GitHub Packages](https://github.com/orgs/hyperonym/packages?repo_name=barba).

### Basic Usage

Starting a CPU-only container:

```bash
docker run -d -p 8501:8501 hyperonym/barba:0.1.0
```

Starting a GPU enabled container:

```bash
docker run --gpus all -d -p 8501:8501 hyperonym/barba:0.1.0
```

#### Natural Language Inference

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
