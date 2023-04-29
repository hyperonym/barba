FROM tensorflow/serving:2.11.1-gpu

# Copy artifacts to the container
COPY servables/barba /models/barba

# Expose ports
EXPOSE 8501

# Provide default environment variables
ENV MODEL_NAME="barba"
ENV TF_CPP_MIN_LOG_LEVEL="2"

# Batching is disabled to support the "cartesian" signature
# Enable batching by adding the following arguments to CMD:
# "--enable_batching=true"
# "--batching_parameters_file=/models/barba/1/assets.extra/batching_parameters.pbtxt"
CMD ["--rest_api_timeout_in_ms=60000", "--enable_model_warmup=true"]
