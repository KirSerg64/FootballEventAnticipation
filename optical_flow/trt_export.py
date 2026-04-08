#!/usr/bin/env python3
"""
TensorRT export and runtime wrapper for SEA-RAFT optical flow.

Overview
--------
The module provides three things:

1. **ONNX export** – wraps the RAFT model so that the ``InputPadder``
   is handled externally and ``iters`` is fixed, enabling
   ``torch.onnx.export`` to unroll the recurrent loop.

2. **TRT engine build** – converts the ONNX file to a TensorRT serialised
   engine with optional FP16 precision and a dynamic-batch optimisation
   profile.

3. **Inference wrapper** – :class:`SeaRaftTRTEngine` mimics the
   ``RAFT.forward`` interface so it can replace the PyTorch model in
   ``_compute_flow_batch`` without any other code changes.

Requirements
------------
    pip install tensorrt onnx
    # CUDA-capable GPU, TensorRT >= 8.6

Usage
-----
Build an engine once (typically 3-10 minutes)::

    from trt_export import build_sea_raft_engine
    build_sea_raft_engine(
        model, args_ns,
        engine_path="sea_raft_360x640_fp16.engine",
        input_h=360, input_w=640,
        max_batch=4, fp16=True,
    )

Then load and use the engine in every subsequent run::

    from trt_export import SeaRaftTRTEngine
    trt_model = SeaRaftTRTEngine("sea_raft_360x640_fp16.engine", device)
    out = trt_model(t1_batch, t2_batch, iters=4, test_mode=True)
    flow = out["flow"][-1]   # [B, 2, H, W]  – same as RAFT.forward
"""

import logging
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_TRT_MIN_VERSION = (8, 6)


# ---------------------------------------------------------------------------
# Padding helpers (replicate InputPadder without SEA-RAFT on sys.path)
# ---------------------------------------------------------------------------

def _compute_pad(h: int, w: int):
    """Return sintel-style padding list [left, right, top, bottom] so that
    both spatial dims become divisible by 8."""
    pad_h = (((h // 8) + 1) * 8 - h) % 8
    pad_w = (((w // 8) + 1) * 8 - w) % 8
    return [pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2]


def _pad_to_mult8(
    image1: torch.Tensor,
    image2: torch.Tensor,
):
    """Pad both images to dimensions divisible by 8.

    Returns ``(image1_padded, image2_padded, pad_spec)`` where *pad_spec*
    is the list ``[left, right, top, bottom]`` needed by :func:`_unpad`.
    """
    h, w = image1.shape[-2:]
    pad = _compute_pad(h, w)
    return (
        F.pad(image1, pad, mode="replicate"),
        F.pad(image2, pad, mode="replicate"),
        pad,
    )


def _unpad(tensor: torch.Tensor, pad_spec) -> torch.Tensor:
    """Remove padding applied by :func:`_pad_to_mult8`."""
    ht, wd = tensor.shape[-2:]
    c = [pad_spec[2], ht - pad_spec[3], pad_spec[0], wd - pad_spec[1]]
    return tensor[..., c[0]: c[1], c[2]: c[3]]


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

class _RaftExportWrapper(nn.Module):
    """Thin wrapper around RAFT that fixes ``iters`` and returns only the
    final flow tensor.

    The ``InputPadder`` inside ``RAFT.forward`` becomes a no-op when the
    caller pre-pads images to multiples of 8 (our contract).  The wrapper
    therefore forwards straight to ``RAFT.forward`` with a fixed Python-int
    ``iters`` so ``torch.onnx.export`` can fully unroll the recurrent loop.
    """

    def __init__(self, raft: nn.Module, fixed_iters: int):
        super().__init__()
        self.raft = raft
        self.fixed_iters = int(fixed_iters)

    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
    ) -> torch.Tensor:
        out = self.raft(image1, image2, iters=self.fixed_iters, test_mode=True)
        return out["flow"][-1]  # [B, 2, H, W]


def export_onnx(
    model: nn.Module,
    args_ns,
    onnx_path: str,
    input_h: int,
    input_w: int,
    batch_size: int = 1,
    max_batch: int = None,
    opset: int = 17,
) -> None:
    """Export SEA-RAFT to ONNX with fixed spatial dimensions and dynamic batch.

    *input_h* and *input_w* must already be divisible by 8 (supply the
    already-padded model-input resolution, not the original video resolution).

    Args:
        model:      Loaded SEA-RAFT model (eval mode, on CUDA/CPU).
        args_ns:    SEA-RAFT config namespace (must contain ``iters``).
        onnx_path:  Destination ``.onnx`` file path.
        input_h:    Model input height, divisible by 8.
        input_w:    Model input width,  divisible by 8.
        batch_size: Representative batch size used to trace the model.
        max_batch:  Upper bound for the dynamic batch axis.  Defaults to
                    *batch_size*.
        opset:      ONNX opset version.  17+ is recommended.
    """
    if input_h % 8 or input_w % 8:
        raise ValueError(
            f"input_h ({input_h}) and input_w ({input_w}) must be divisible by 8"
        )

    fixed_iters = int(getattr(args_ns, "iters", 4))
    if max_batch is None:
        max_batch = batch_size

    # ── ONNX export strategy ──────────────────────────────────────────────
    # Problem history:
    #   1. CUDA + dynamo=True (default) → UnsupportedOperatorException:
    #      aten.cudnn_grid_sampler (CUDA grid_sample can't be lowered by dynamo)
    #   2. CPU + ScriptModule + dynamo=True → ValueError: ScriptModule not supported
    #   3. CPU + plain module + dynamo=True + dynamic_shapes →
    #      TRT ONNX parser fails to import initializers (dynamo packs weights
    #      in formats TRT < 10 can't parse, e.g. non-standard dtype metadata)
    #
    # Correct approach: dynamo=False + plain nn.Module + CPU
    #   • dynamo=False: legacy TorchScript-trace path – initializers stored as
    #     plain float32 ONNX tensors, always parseable by TRT
    #   • plain nn.Module (not pre-traced): ScriptModule rejection only happens
    #     when we pass a jit.trace result to the dynamo path
    #   • CPU: F.grid_sample → aten.grid_sampler (ONNX-compatible) instead of
    #     aten.cudnn_grid_sampler
    #   • dynamic_axes: the correct dynamic-batch parameter for the legacy path
    #     (dynamic_shapes is the dynamo-path parameter; they are mutually exclusive)
    orig_dev = next(model.parameters()).device
    try:
        wrapper = _RaftExportWrapper(model.cpu(), fixed_iters).eval()
        # Use a realistic mid-gray dummy (127.5) instead of all-zeros.
        # RAFT.forward normalizes by 2*(x/255)-1; all-zero inputs produce constant
        # activation maps that may activate different code paths during TorchScript
        # tracing (e.g. epsilon guards in correlation / normalisation layers).
        dummy = torch.full(
            (batch_size, 3, input_h, input_w), 127.5, dtype=torch.float32
        )

        dynamic_axes = {
            "image1": {0: "batch"},
            "image2": {0: "batch"},
            "flow":   {0: "batch"},
        }

        logger.info(
            "Exporting ONNX (legacy TorchScript, CPU) → %s  "
            "(opset=%d, H=%d, W=%d, iters=%d, batch=%d, max_batch=%d)",
            onnx_path, opset, input_h, input_w, fixed_iters, batch_size, max_batch,
        )
        print(
            f"[INFO] Exporting ONNX (iters={fixed_iters}, "
            f"H={input_h}, W={input_w}, max_batch={max_batch}) → {onnx_path}"
        )
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy, dummy),
                onnx_path,
                input_names=["image1", "image2"],
                output_names=["flow"],
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                do_constant_folding=True,
                dynamo=False,   # legacy path: plain float32 initializers, TRT-compatible
            )
    finally:
        model.to(orig_dev)

    print(f"[INFO] ONNX export complete: {onnx_path}")


# ---------------------------------------------------------------------------
# TensorRT engine build
# ---------------------------------------------------------------------------

def _check_trt_version(trt) -> None:
    parts = tuple(int(x) for x in trt.__version__.split(".")[:2])
    if parts < _TRT_MIN_VERSION:
        raise RuntimeError(
            f"TensorRT {trt.__version__} is too old; "
            f">= {'.'.join(str(v) for v in _TRT_MIN_VERSION)} required."
        )


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_gb: int = 4,
    max_batch: int = 1,
    input_h: int = None,
    input_w: int = None,
) -> None:
    """Build a TensorRT serialised engine from an ONNX file.

    Requires TensorRT >= 8.6 and a CUDA-capable GPU.  Building typically
    takes 3-10 minutes for RAFT-sized models.

    Args:
        onnx_path:    Input ONNX file.
        engine_path:  Output ``.engine`` file.
        fp16:         Enable FP16 (recommended; cuts latency roughly in half).
        workspace_gb: TRT builder scratch space in GiB.
        max_batch:    Maximum batch size to include in the optimisation profile.
        input_h:      Spatial height for the optimisation profile.  Inferred
                      from the ONNX model when ``None``.
        input_w:      Spatial width.  Inferred when ``None``.
    """
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError(
            "TensorRT Python package not found.  "
            "Install it from https://developer.nvidia.com/tensorrt "
            "or via:  pip install tensorrt"
        ) from exc

    _check_trt_version(trt)

    # ── infer spatial dims from ONNX if not supplied ──────────────────────
    if input_h is None or input_w is None:
        try:
            import onnx
            proto = onnx.load(onnx_path)
            inp = proto.graph.input[0]
            dims = inp.type.tensor_type.shape.dim
            # shape: [batch, 3, H, W]
            inferred_h = dims[2].dim_value
            inferred_w = dims[3].dim_value
            if inferred_h == 0 or inferred_w == 0:
                raise ValueError("ONNX has dynamic spatial dims; pass input_h/input_w explicitly.")
            input_h = inferred_h
            input_w = inferred_w
        except ImportError:
            raise RuntimeError(
                "Cannot infer input_h/input_w: install onnx (pip install onnx) "
                "or pass them explicitly."
            )

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder    = trt.Builder(trt_logger)
    network    = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        raw = f.read()
    if not parser.parse(raw):
        errors = "\n".join(
            str(parser.get_error(i)) for i in range(parser.num_errors)
        )
        raise RuntimeError(f"ONNX parse errors:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1024 ** 3))
    )

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] TRT build: FP16 enabled.")
        else:
            print("[WARN] Platform does not support fast FP16; building FP32 engine.")
    else:
        print("[INFO] TRT build: FP32 only.")

    # ── optimisation profile ──────────────────────────────────────────────
    profile = builder.create_optimization_profile()
    opt_b   = max(1, max_batch // 2)
    for name in ("image1", "image2"):
        profile.set_shape(
            name,
            min=(1,       3, input_h, input_w),
            opt=(opt_b,   3, input_h, input_w),
            max=(max_batch, 3, input_h, input_w),
        )
    config.add_optimization_profile(profile)

    print(
        f"[INFO] Building TRT engine "
        f"(H={input_h}, W={input_w}, max_batch={max_batch}, "
        f"workspace={workspace_gb} GiB).  This may take several minutes …"
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(
            "TRT engine build failed: build_serialized_network returned None."
        )

    with open(engine_path, "wb") as f:
        f.write(serialized)
    size_mb = serialized.size / 1e6
    print(f"[INFO] TRT engine saved → {engine_path}  ({size_mb:.1f} MB)")


def build_sea_raft_engine(
    model: nn.Module,
    args_ns,
    engine_path: str,
    input_h: int = 360,
    input_w: int = 640,
    max_batch: int = 1,
    fp16: bool = True,
    workspace_gb: int = 4,
    opset: int = 17,
) -> None:
    """Export to ONNX then build a TRT engine in one call.

    The intermediate ONNX file is written to a temporary location and
    removed automatically afterwards.

    Args:
        model:        Loaded SEA-RAFT model (eval mode, on CUDA).
        args_ns:      SEA-RAFT config namespace.
        engine_path:  Destination ``.engine`` file.
        input_h:      Model input height after scale (must be divisible by 8).
                      For 720p video with ``scale=-1`` this is **360**.
        input_w:      Model input width.  For 720p / ``scale=-1``: **640**.
        max_batch:    Maximum batch size for the TRT optimisation profile.
        fp16:         Build with FP16 precision.
        workspace_gb: TRT builder workspace in GiB.
        opset:        ONNX opset version.
    """
    os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        onnx_path = tmp.name
    try:
        export_onnx(
            model, args_ns, onnx_path,
            input_h=input_h, input_w=input_w,
            batch_size=max_batch, max_batch=max_batch, opset=opset,
        )
        build_engine(
            onnx_path, engine_path,
            fp16=fp16, workspace_gb=workspace_gb,
            max_batch=max_batch, input_h=input_h, input_w=input_w,
        )
    finally:
        if os.path.isfile(onnx_path):
            os.remove(onnx_path)


# ---------------------------------------------------------------------------
# TRT inference engine
# ---------------------------------------------------------------------------

class SeaRaftTRTEngine:
    """Drop-in replacement for a loaded RAFT model backed by a TRT engine.

    The call signature matches ``RAFT.forward(image1, image2, iters=N,
    test_mode=True)`` – the ``iters`` argument is ignored at runtime because
    it was baked in at engine build time.

    The caller passes float32 tensors in ``[0, 255]`` range at the scaled
    spatial resolution (e.g. ``[B, 3, 360, 640]`` for 720p with
    ``scale=-1``).  Internal padding to the nearest multiple of 8 is handled
    by the engine wrapper.

    Requirements
    ------------
    TensorRT >= 8.6, CUDA, engine built by :func:`build_sea_raft_engine`.
    """

    def __init__(self, engine_path: str, device: torch.device):
        """
        Args:
            engine_path: Path to the serialised ``.engine`` file.
            device:      CUDA device to run inference on.
        """
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                "TensorRT Python package not found.  "
                "Install it from https://developer.nvidia.com/tensorrt "
                "or via:  pip install tensorrt"
            ) from exc

        _check_trt_version(trt)

        self.device = device
        self._trt = trt  # keep reference for later use

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt_logger)
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(
                f"Failed to deserialise TRT engine from {engine_path}"
            )
        self._context = self._engine.create_execution_context()

        # Dedicated CUDA stream so inference doesn't block the default stream.
        self._stream = torch.cuda.Stream(device=device)

        # Discover all IO tensor names and which are inputs/outputs.
        n = self._engine.num_io_tensors
        self._io_names   = [self._engine.get_tensor_name(i) for i in range(n)]
        self._input_names  = [
            nm for nm in self._io_names
            if self._engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT
        ]
        self._output_names = [
            nm for nm in self._io_names
            if self._engine.get_tensor_mode(nm) == trt.TensorIOMode.OUTPUT
        ]

        # Cache output dtype so __call__ allocates a matching buffer.
        # TRT compiles outputs as HALF when FP16 mode is active; writing FP16
        # bytes into a float32 buffer causes every two FP16 values to be
        # misread as one float32 — exactly the "batch mixing" symptom.
        _trt_to_torch = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF:  torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8:  torch.int8,
        }
        self._out_dtype = _trt_to_torch.get(
            self._engine.get_tensor_dtype("flow"), torch.float32
        )
        logger.info(
            "TRT engine loaded from %s  |  inputs: %s  outputs: %s",
            engine_path, self._input_names, self._output_names,
        )
        print(
            f"[INFO] TRT engine loaded: {engine_path}  "
            f"(inputs={self._input_names}, outputs={self._output_names})"
        )

    # ── forward ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def __call__(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        iters: int = None,   # ignored – baked in at build time
        test_mode: bool = True,
        flow_gt=None,        # ignored – training-only
    ) -> dict:
        """Run TRT inference.

        Returns a dict with the same keys as ``RAFT.forward`` in test mode::

            {"flow": [flow_tensor],  "info": [],  "nf": None}

        where ``flow_tensor`` has shape ``[B, 2, H, W]`` at the input
        spatial resolution (padding is applied internally then stripped).
        """
        # ── pad to multiples of 8 ─────────────────────────────────────────
        img1_pad, img2_pad, pad_spec = _pad_to_mult8(image1, image2)
        B, _, H_pad, W_pad = img1_pad.shape

        img1_pad = img1_pad.contiguous().to(self.device, dtype=torch.float32)
        img2_pad = img2_pad.contiguous().to(self.device, dtype=torch.float32)

        # ── set dynamic input shapes in context ───────────────────────────
        self._context.set_input_shape("image1", (B, 3, H_pad, W_pad))
        self._context.set_input_shape("image2", (B, 3, H_pad, W_pad))

        # ── allocate output buffer ─────────────────────────────────────────
        # Allocate with the engine's actual output dtype (may be float16 when
        # the engine was built with FP16).  Dtype mismatch here causes TRT to
        # write N-byte values into a 2N-byte buffer, corrupting all batch items.
        flow_buf = torch.empty(B, 2, H_pad, W_pad, dtype=self._out_dtype, device=self.device)

        # ── bind tensor addresses ─────────────────────────────────────────
        self._context.set_tensor_address("image1", img1_pad.data_ptr())
        self._context.set_tensor_address("image2", img2_pad.data_ptr())
        self._context.set_tensor_address("flow",   flow_buf.data_ptr())

        # ── asynchronous execution ────────────────────────────────────────
        # img1_pad / img2_pad were prepared on the default CUDA stream (F.pad,
        # .to(), .contiguous()).  execute_async_v3 runs on self._stream.  Without
        # an explicit dependency, both streams run concurrently and TRT may read
        # the input buffers before the default stream has finished writing them —
        # causing garbled / stale flow maps.  Force self._stream to wait first.
        self._stream.wait_stream(torch.cuda.current_stream(self.device))
        self._context.execute_async_v3(self._stream.cuda_stream)
        torch.cuda.current_stream(self.device).wait_stream(self._stream)

        # ── remove padding and normalise dtype to float32 ─────────────────
        flow = _unpad(flow_buf.float(), pad_spec)

        return {"flow": [flow], "info": [], "nf": None}
