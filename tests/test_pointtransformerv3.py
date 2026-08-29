import pytest
import torch

from src.pointtransformerv3_model import (
    PTV3_AMP_DTYPE,
    PTV3_SPCONV_ALGORITHM,
    PTV3_UPSTREAM_REVISION,
    PointTransformerV3Encoder,
    default_pointtransformerv3_config,
    deterministic_voxelize,
    validate_pointtransformerv3_checkpoint_config,
)
from src.precision import (
    checkpoint_amp_dtype,
    grad_scaler_enabled,
    resolve_amp_dtype,
)
from train_pipeline import build_parser, landmark_model_config


def _voxel_input():
    return torch.tensor(
        [
            [
                [0.01, 0.01, 0.01, 1.0, 0.0, 0.0],
                [0.02, 0.02, 0.02, 2.0, 0.0, 0.0],
                [0.21, 0.01, 0.01, 3.0, 0.0, 0.0],
            ],
            [
                [-0.21, 0.01, 0.01, 4.0, 0.0, 0.0],
                [-0.19, 0.01, 0.01, 5.0, 0.0, 0.0],
                [0.11, 0.01, 0.01, 6.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )


def test_ptv3_voxelization_is_deterministic_unique_and_batched():
    points = _voxel_input()
    first, first_counts = deterministic_voxelize(points, grid_size=0.1)
    second, second_counts = deterministic_voxelize(points, grid_size=0.1)

    assert first_counts == second_counts == [2, 3]
    assert first["offset"].tolist() == [2, 5]
    assert first["batch"].tolist() == [0, 0, 1, 1, 1]
    for key in first:
        assert torch.equal(first[key], second[key])
    sparse_indices = torch.cat(
        [first["batch"].unsqueeze(1).to(torch.int32), first["grid_coord"]], dim=1
    )
    assert torch.unique(sparse_indices, dim=0).shape[0] == sparse_indices.shape[0]
    assert first["feat"][:, 3].tolist() == [1.0, 3.0, 4.0, 5.0, 6.0]


def test_ptv3_voxelization_preserves_gradients_only_for_representatives():
    points = _voxel_input()[:1].clone().requires_grad_(True)
    data, counts = deterministic_voxelize(points, grid_size=0.1)
    data["feat"].sum().backward()

    assert counts == [2]
    assert points.grad is not None
    assert torch.count_nonzero(points.grad[0, 0]) > 0
    assert torch.count_nonzero(points.grad[0, 1]) == 0
    assert torch.count_nonzero(points.grad[0, 2]) > 0


def test_ptv3_locked_default_configuration():
    config = default_pointtransformerv3_config(0.02)

    assert config["grid_size"] == pytest.approx(0.02)
    assert config["upstream_revision"] == PTV3_UPSTREAM_REVISION
    assert config["enable_flash"] is True
    assert config["enc_depths"] == [2, 2, 2, 6, 2]
    assert config["dec_depths"] == [2, 2, 2, 2]
    assert config["dec_channels"] == [64, 64, 128, 256]
    assert config["global_pool"] == "max"
    assert config["voxel_representative"] == "first_input_index"
    assert config["order_shuffle_policy"] == "training_only"
    assert config["spconv_algorithm"] == PTV3_SPCONV_ALGORITHM


def test_ptv3_rejects_non_flash_or_changed_adapter_contract_before_imports():
    with pytest.raises(ValueError, match="requires FlashAttention"):
        PointTransformerV3Encoder(enable_flash=False)
    with pytest.raises(ValueError, match="global max pooling"):
        PointTransformerV3Encoder(global_pool="mean")
    with pytest.raises(ValueError, match="training-only order shuffling"):
        PointTransformerV3Encoder(order_shuffle_policy="always")
    with pytest.raises(ValueError, match="spconv_algorithm='native'"):
        PointTransformerV3Encoder(spconv_algorithm="mask_implicit_gemm")


def test_ptv3_serialization_order_shuffle_is_training_only():
    encoder = PointTransformerV3Encoder.__new__(PointTransformerV3Encoder)
    torch.nn.Module.__init__(encoder)
    root = torch.nn.Module()
    root.shuffle_orders = True
    child = torch.nn.Module()
    child.shuffle_orders = True
    root.add_module("pool", child)
    encoder.model = root

    encoder.eval()
    encoder._set_order_shuffle()
    assert root.shuffle_orders is False
    assert child.shuffle_orders is False

    encoder.train()
    encoder._set_order_shuffle()
    assert root.shuffle_orders is True
    assert child.shuffle_orders is True


def test_ptv3_cli_records_grid_size_and_preflight_defaults():
    parser = build_parser()
    training = parser.parse_args(
        [
            "fit-landmarks",
            "--folds-json",
            "folds.json",
            "--outer-fold",
            "0",
            "--predictions-json",
            "predictions.json",
            "--calibration-json",
            "calibration.json",
            "--output-dir",
            "runs/ptv3",
            "--backbone",
            "pointtransformerv3",
            "--ptv3-grid-size",
            "0.02",
            "--learning-rate",
            "0.001",
            "--encoder-learning-rate",
            "0.0003",
            "--weight-decay",
            "0.01",
            "--warmup-epochs",
            "10",
            "--minimum-learning-rate",
            "0.000001",
            "--gradient-clip-norm",
            "1.0",
            "--effective-batch-size",
            "32",
        ]
    )
    preflight = parser.parse_args(["ptv3-preflight"])

    assert training.backbone == "pointtransformerv3"
    assert training.ptv3_grid_size == pytest.approx(0.02)
    assert training.learning_rate == pytest.approx(1e-3)
    assert training.encoder_learning_rate == pytest.approx(3e-4)
    assert training.weight_decay == pytest.approx(1e-2)
    assert training.warmup_epochs == 10
    assert training.minimum_learning_rate == pytest.approx(1e-6)
    assert training.gradient_clip_norm == pytest.approx(1.0)
    assert training.effective_batch_size == 32
    model_config = landmark_model_config(training, local_scale=40.0)
    assert model_config["amp_dtype"] == PTV3_AMP_DTYPE
    assert (
        model_config["encoder_config"]["spconv_algorithm"]
        == PTV3_SPCONV_ALGORITHM
    )
    assert preflight.grid_size == [0.01, 0.02]
    assert preflight.num_points == 16384

    dense_preflight = parser.parse_args(
        [
            "ptv3-preflight",
            "--num-points",
            "32768",
            "--grid-size",
            "0.02",
        ]
    )
    assert dense_preflight.num_points == 32768
    assert dense_preflight.grid_size == [0.02]


def test_ptv3_fp16_policy_is_explicit_and_uses_gradient_scaling():
    cuda = torch.device("cuda")

    assert resolve_amp_dtype(cuda, True, "float16") is torch.float16
    assert grad_scaler_enabled(cuda, True, "float16") is True
    assert checkpoint_amp_dtype({"amp_dtype": "float16"}) == "float16"
    assert checkpoint_amp_dtype({}) == "none"


def test_ptv3_checkpoint_requires_native_spconv_workaround():
    config = {
        "backbone": "pointtransformerv3",
        "amp_dtype": PTV3_AMP_DTYPE,
        "encoder_config": default_pointtransformerv3_config(0.01),
    }
    validate_pointtransformerv3_checkpoint_config(config)

    invalid = {**config, "encoder_config": dict(config["encoder_config"])}
    invalid["encoder_config"]["spconv_algorithm"] = "mask_implicit_gemm"
    with pytest.raises(ValueError, match="spconv_algorithm='native'"):
        validate_pointtransformerv3_checkpoint_config(invalid)


def test_ptv3_rejects_disabling_amp():
    parser = build_parser()
    training = parser.parse_args(
        [
            "fit-landmarks",
            "--folds-json",
            "folds.json",
            "--outer-fold",
            "0",
            "--predictions-json",
            "predictions.json",
            "--calibration-json",
            "calibration.json",
            "--output-dir",
            "runs/ptv3",
            "--backbone",
            "pointtransformerv3",
            "--no-amp",
        ]
    )

    with pytest.raises(ValueError, match="requires FP16 AMP"):
        landmark_model_config(training, local_scale=40.0)


def test_auto_amp_policy_for_other_backbones_is_unchanged(monkeypatch):
    cuda = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_amp_dtype(cuda, True, "auto") is torch.bfloat16
    assert grad_scaler_enabled(cuda, True, "auto") is False

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert resolve_amp_dtype(cuda, True, "auto") is torch.float16
    assert grad_scaler_enabled(cuda, True, "auto") is True
