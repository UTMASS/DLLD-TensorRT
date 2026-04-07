"""
TensorRT inference backend for UFLD v2.

Drop-in replacement for onnxruntime.InferenceSession — provides the same
`run()` interface so the detector class can switch backends with minimal changes.

Uses PyTorch for GPU memory management (no PyCUDA / CUDA Toolkit needed).

Requirements:
  pip install tensorrt-cu12 torch numpy
"""

import numpy as np
import torch
import tensorrt as trt


# Map TensorRT dtypes to (numpy dtype, torch dtype)
_TRT_DTYPE_MAP = {
    trt.float32: (np.float32, torch.float32),
    trt.float16: (np.float16, torch.float16),
    trt.int32:   (np.int32,   torch.int32),
    trt.int8:    (np.int8,    torch.int8),
    trt.bool:    (np.bool_,   torch.bool),
}

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTBackend:
    """
    TensorRT inference backend using PyTorch CUDA tensors for memory.

    Usage:
        backend = TRTBackend("culane_res18_fp16.trt")
        outputs = backend.run(input_array)  # input_array: np.ndarray NCHW
    """

    def __init__(self, engine_path):
        """Load a serialized TensorRT engine and allocate GPU buffers."""
        print(f"[TRT] Loading engine: {engine_path}")

        # Deserialize engine
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Discover bindings and allocate PyTorch CUDA tensors
        self.input_binding = None
        self.output_bindings = []
        self._input_name = None
        self._input_shape = None
        self._input_np_dtype = None

        self._d_tensors = {}   # GPU tensors {name: torch.Tensor on CUDA}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            trt_dtype = self.engine.get_tensor_dtype(name)
            np_dtype, torch_dtype = _TRT_DTYPE_MAP.get(trt_dtype, (np.float32, torch.float32))
            mode = self.engine.get_tensor_mode(name)

            # Allocate GPU tensor
            self._d_tensors[name] = torch.empty(shape, dtype=torch_dtype, device="cuda")

            if mode == trt.TensorIOMode.INPUT:
                self.input_binding = name
                self._input_name = name
                self._input_shape = shape
                self._input_np_dtype = np_dtype
                print(f"[TRT]   Input  '{name}': {shape} {np.dtype(np_dtype).name}")
            else:
                self.output_bindings.append(name)
                print(f"[TRT]   Output '{name}': {shape} {np.dtype(np_dtype).name}")

        # CUDA stream
        self.stream = torch.cuda.Stream()

        print(f"[TRT] Engine ready ({len(self.output_bindings)} outputs)")

    @property
    def input_name(self):
        return self._input_name

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def input_dtype(self):
        return self._input_np_dtype

    def run(self, input_array):
        """
        Run inference.

        Args:
            input_array: np.ndarray with shape matching engine input (e.g., [1,3,320,1600])

        Returns:
            List of np.ndarray outputs (same order as ONNX model outputs).
        """
        with torch.cuda.stream(self.stream):
            # H2D: copy numpy input into GPU tensor
            inp_tensor = self._d_tensors[self.input_binding]
            inp_tensor.copy_(torch.from_numpy(np.ascontiguousarray(input_array)).to(inp_tensor.dtype))

            # Set tensor addresses (raw GPU pointers)
            for name, tensor in self._d_tensors.items():
                self.context.set_tensor_address(name, tensor.data_ptr())

            # Execute inference
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)

        # Sync
        self.stream.synchronize()

        # D2H: copy outputs back to numpy
        return [self._d_tensors[name].cpu().numpy() for name in self.output_bindings]

    def close(self):
        """Free GPU resources."""
        del self.context
        del self.engine
        self._d_tensors.clear()
        torch.cuda.empty_cache()


class TRTBackendONNXCompat:
    """
    Wrapper that mimics onnxruntime.InferenceSession interface for drop-in use.

    Usage:
        session = TRTBackendONNXCompat("engine.trt")
        outputs = session.run(None, {"input": blob})
    """

    def __init__(self, engine_path):
        self._backend = TRTBackend(engine_path)

        # Mimic onnxruntime input metadata
        class _InputMeta:
            def __init__(self, name, shape, dtype):
                self.name = name
                self.shape = list(shape)
                self.type = "tensor(float16)" if dtype == np.float16 else "tensor(float)"

        self._inputs = [_InputMeta(
            self._backend.input_name,
            self._backend.input_shape,
            self._backend.input_dtype
        )]

    def get_inputs(self):
        return self._inputs

    def get_providers(self):
        return ["TensorrtExecutionProvider"]

    def run(self, output_names, input_dict):
        """Match onnxruntime session.run(None, {input_name: blob}) signature."""
        input_array = list(input_dict.values())[0]
        return self._backend.run(input_array)

    def close(self):
        self._backend.close()
