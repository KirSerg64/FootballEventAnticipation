#!/usr/bin/env python3
"""
Unit tests for FUTR.forward using fully synthetic data.

Branches covered
----------------
- train mode vs. eval (validation) mode
- anticipation head (action + offset outputs)
- segmentation head
- actionness head
- all heads simultaneously
- encoder-only mode  (num_decoder_layers=0)
- padding mask applied to padded label sequences
- optical flow branch – documents current implementation limitation

No datasets, checkpoints, or pretrained weights are required.
timm.create_model is patched with a minimal fake backbone.
"""

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from model.futr import FUTR

# ---------------------------------------------------------------------------
# Synthetic-data constants
# ---------------------------------------------------------------------------
_B = 2          # batch size
_S = 4          # observed sequence length
_N_CLASS = 5    # number of classes (including background=0)
_N_QUERY = 3    # anticipation queries
_HIDDEN = 32    # transformer hidden dim
_N_HEAD = 4     # attention heads
_FEAT_DIM = 64  # fake backbone output dim  (must be divisible by _N_HEAD)
_H, _W = 32, 32 # spatial frame size
_PAD_IDX = -1


# ---------------------------------------------------------------------------
# Minimal fake backbone – replaces timm to avoid downloading pretrained weights
# ---------------------------------------------------------------------------

class _FakeFC:
    """Stand-in for backbone.head.fc – only 'in_features' is read."""
    in_features = _FEAT_DIM


class _FakeBackbone(nn.Module):
    """Returns zero feature vectors; accepts any spatial input size."""

    def __init__(self):
        super().__init__()
        self.head = SimpleNamespace(fc=_FakeFC())

    def forward(self, x):           # x: [N, C, H, W]
        return torch.zeros(x.shape[0], _FEAT_DIM, device=x.device)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_args(*, seg=False, anticipate=True, actionness=False, jointtrain=None):
    return argparse.Namespace(
        feature_arch='rny004',          # no _gsf / _gsm → no temporal-shift wrap
        temporal_arch='none',
        n_layers=2,
        sgp_ks=3,
        sgp_r=2,
        clip_len=8,
        cheating_dataset=False,
        cheating_range=[0.0, 1.0],
        obs_perc=[0.5],
        seg=seg,
        anticipate=anticipate,
        actionness=actionness,
        pos_emb=True,
        max_pos_len=200,
        jointtrain=jointtrain,
    )


def _build_model(
    args,
    *,
    num_encoder_layers=1,
    num_decoder_layers=1,
    n_query=_N_QUERY,
    use_optical_flow=False,
):
    with patch('model.futr.timm.create_model', return_value=_FakeBackbone()):
        model = FUTR(
            n_class=_N_CLASS,
            hidden_dim=_HIDDEN,
            src_pad_idx=_PAD_IDX,
            device=torch.device('cpu'),
            args=args,
            n_query=n_query,
            n_head=_N_HEAD,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            use_optical_flow=use_optical_flow,
        )
    model.eval()
    return model


def _frames(b=_B, s=_S):
    """Synthetic [B, S, 3, H, W] uint8-range float frames."""
    return torch.randint(0, 256, (b, s, 3, _H, _W), dtype=torch.float32)


def _label(b=_B, s=_S):
    """Non-padded past-label tensor [B, S]."""
    return torch.zeros(b, s, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Output-key presence
# ---------------------------------------------------------------------------

class TestOutputKeys:

    def test_anticipate_only_keys(self):
        model = _build_model(_make_args(anticipate=True, seg=False, actionness=False))
        with torch.no_grad():
            out = model((_frames(), _label()), mode='train')
        assert set(out) == {'action', 'offset'}

    def test_seg_only_keys(self):
        model = _build_model(_make_args(anticipate=False, seg=True, actionness=False))
        with torch.no_grad():
            out = model((_frames(), _label()), mode='train')
        assert set(out) == {'seg'}

    def test_actionness_adds_key(self):
        model = _build_model(_make_args(anticipate=True, seg=False, actionness=True))
        with torch.no_grad():
            out = model((_frames(), _label()), mode='train')
        assert 'actionness' in out

    def test_all_heads_present(self):
        model = _build_model(_make_args(anticipate=True, seg=True, actionness=True))
        with torch.no_grad():
            out = model((_frames(), _label()), mode='train')
        assert {'action', 'offset', 'seg', 'actionness'} == set(out)


# ---------------------------------------------------------------------------
# Output tensor shapes
# ---------------------------------------------------------------------------

class TestOutputShapes:

    def test_anticipation_shapes(self):
        model = _build_model(_make_args(anticipate=True, seg=False, actionness=False))
        with torch.no_grad():
            out = model((_frames(), _label()), mode='train')
        # action: [B, n_query, n_class]
        assert out['action'].shape == (_B, _N_QUERY, _N_CLASS)
        # offset: [B, n_query]
        assert out['offset'].shape == (_B, _N_QUERY)

    def test_anticipation_with_actionness_reduces_class_dim(self):
        """With actionness the action head omits the EOS class → n_class-1."""
        model = _build_model(_make_args(anticipate=True, seg=False, actionness=True))
        with torch.no_grad():
            out = model((_frames(), _label()), mode='train')
        assert out['action'].shape == (_B, _N_QUERY, _N_CLASS - 1)

    def test_seg_shape(self):
        model = _build_model(_make_args(anticipate=False, seg=True, actionness=False))
        with torch.no_grad():
            out = model((_frames(), _label()), mode='train')
        # seg: [B, S, n_class]
        assert out['seg'].shape == (_B, _S, _N_CLASS)

    def test_actionness_shape(self):
        model = _build_model(_make_args(anticipate=True, seg=False, actionness=True))
        with torch.no_grad():
            out = model((_frames(), _label()), mode='train')
        assert out['actionness'].shape == (_B, _N_QUERY)

    def test_shapes_all_heads(self):
        model = _build_model(_make_args(anticipate=True, seg=True, actionness=True))
        with torch.no_grad():
            out = model((_frames(), _label()), mode='train')
        assert out['action'].shape == (_B, _N_QUERY, _N_CLASS - 1)
        assert out['offset'].shape == (_B, _N_QUERY)
        assert out['seg'].shape == (_B, _S, _N_CLASS)
        assert out['actionness'].shape == (_B, _N_QUERY)


# ---------------------------------------------------------------------------
# Eval / validation mode
# ---------------------------------------------------------------------------

class TestEvalMode:

    def test_eval_does_not_require_label(self):
        """In validation mode inputs is just src – no label tensor."""
        model = _build_model(_make_args(anticipate=True, seg=False, actionness=False))
        with torch.no_grad():
            out = model(_frames(), mode='validation')
        assert 'action' in out and 'offset' in out

    def test_eval_anticipation_shapes(self):
        model = _build_model(_make_args(anticipate=True, seg=False, actionness=False))
        with torch.no_grad():
            out = model(_frames(), mode='validation')
        assert out['action'].shape == (_B, _N_QUERY, _N_CLASS)
        assert out['offset'].shape == (_B, _N_QUERY)

    def test_eval_seg_shape(self):
        model = _build_model(_make_args(anticipate=False, seg=True, actionness=False))
        with torch.no_grad():
            out = model(_frames(), mode='validation')
        assert out['seg'].shape == (_B, _S, _N_CLASS)

    def test_eval_all_heads_shapes(self):
        model = _build_model(_make_args(anticipate=True, seg=True, actionness=True))
        with torch.no_grad():
            out = model(_frames(), mode='validation')
        assert out['action'].shape == (_B, _N_QUERY, _N_CLASS - 1)
        assert out['seg'].shape == (_B, _S, _N_CLASS)
        assert out['actionness'].shape == (_B, _N_QUERY)


# ---------------------------------------------------------------------------
# Encoder-only mode  (num_decoder_layers=0)
# ---------------------------------------------------------------------------

class TestEncoderOnly:

    def test_encoder_only_requires_n_query_1(self):
        """num_decoder_layers=0 with n_query>1 must raise ValueError."""
        args = _make_args(anticipate=True, seg=False, actionness=False)
        with pytest.raises(ValueError):
            _build_model(args, num_decoder_layers=0, n_query=_N_QUERY)

    def test_encoder_only_with_n_query_1_runs(self):
        args = _make_args(anticipate=True, seg=False, actionness=False)
        model = _build_model(args, num_decoder_layers=0, n_query=1)
        with torch.no_grad():
            out = model((_frames(), _label()), mode='train')
        assert 'action' in out

    def test_encoder_only_eval_mode(self):
        args = _make_args(anticipate=True, seg=False, actionness=False)
        model = _build_model(args, num_decoder_layers=0, n_query=1)
        with torch.no_grad():
            out = model(_frames(), mode='validation')
        assert 'action' in out


# ---------------------------------------------------------------------------
# Padding mask
# ---------------------------------------------------------------------------

class TestPaddingMask:

    def test_fully_padded_last_timestep_does_not_raise(self):
        """Sequences with some padding positions should be handled without error."""
        model = _build_model(_make_args(anticipate=True, seg=False, actionness=False))
        src_label = _label()
        src_label[:, -1] = _PAD_IDX       # mark last frame as padding
        with torch.no_grad():
            out = model((_frames(), src_label), mode='train')
        assert 'action' in out

    def test_all_padded_does_not_raise(self):
        """Fully-padded batch is pathological but must not crash."""
        model = _build_model(_make_args(anticipate=True, seg=False, actionness=False))
        src_label = torch.full((_B, _S), _PAD_IDX, dtype=torch.float32)
        with torch.no_grad():
            out = model((_frames(), src_label), mode='train')
        assert 'action' in out


# ---------------------------------------------------------------------------
# Optical flow branch
#
# NOTE: The current forward implementation concatenates the flow tensor
# spatially (dim=2) *before* passing frames to the backbone, producing a
# 5-channel input to a backbone that expects 3-channel images.
# The `standarize` step then applies ImageNet normalisation over 3 channels
# to a 5-channel tensor (or the backbone rejects the channel count).
# These tests document the known behaviour: the optical-flow branch raises
# an exception until the integration is corrected (e.g. by extracting flow
# features separately and concatenating in feature space after the backbone).
# ---------------------------------------------------------------------------

class TestOpticalFlowBranch:

    def _flow(self, b=_B, s=_S):
        return torch.zeros(b, s, 2, _H, _W)

    def test_train_flow_raises_with_current_implementation(self):
        args = _make_args(anticipate=True, seg=False, actionness=False)
        model = _build_model(args, use_optical_flow=True)
        inputs = (_frames(), _label(), self._flow())
        with pytest.raises(Exception):
            model(inputs, mode='train')

    def test_eval_flow_raises_with_current_implementation(self):
        args = _make_args(anticipate=True, seg=False, actionness=False)
        model = _build_model(args, use_optical_flow=True)
        inputs = (_frames(), self._flow())
        with pytest.raises(Exception):
            model(inputs, mode='validation')
