import tensorflow as _tf

gpus = _tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across _GPUs
        for _gpu in gpus:
            _tf.config.experimental.set_memory_growth(_gpu, True)
        _logical_gpus = _tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical _GPUs,", len(_logical_gpus), "Logical _GPUs")
    except RuntimeError as _e:
        # Memory growth must be set before _GPUs have been initialized
        print(_e)


