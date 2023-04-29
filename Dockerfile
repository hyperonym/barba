FROM python:3.10.11 AS builder

# Create builder directory
WORKDIR /builder

# Install builder dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Bundle builder source
COPY . .

# Build servable model
RUN python download.py hyperonym/barba models/barba
RUN python export.py models/barba servables/barba/1

FROM tensorflow/serving:2.11.1-gpu

# Copy artifacts to the container
COPY --from=builder /builder/servables/barba /models/barba

# Expose ports
EXPOSE 8501

# Provide default environment variables
ENV MODEL_NAME="barba"
ENV TF_CPP_MIN_LOG_LEVEL="2"

# Enable batching by adding the following arguments to CMD:
# "--enable_batching=true"
# "--batching_parameters_file=/models/barba/1/assets.extra/batching_parameters.pbtxt"
CMD ["--rest_api_timeout_in_ms=60000", "--enable_model_warmup=true"]
