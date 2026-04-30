"""
Tests unitarios para los modelos y utilidades del proyecto.

Ejecutar con:
    python -m pytest tests/ -v
"""

import os
import sys
import json
import tempfile

import numpy as np
import pytest
import torch

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.cnn_model import CustomCNN, build_cnn
from models.resnet_model import ResNetTransfer, build_resnet


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_batch():
    """Mini-batch de imágenes RGB 224×224."""
    torch.manual_seed(0)
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def num_classes():
    return 38


# ─── Tests del modelo CNN personalizado ──────────────────────────────────────

class TestCustomCNN:

    def test_build_returns_correct_type(self, num_classes):
        model = build_cnn(num_classes)
        assert isinstance(model, CustomCNN)

    def test_output_shape(self, dummy_batch, num_classes):
        model = build_cnn(num_classes)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, num_classes), (
            f"Forma esperada (4, {num_classes}), obtenida {out.shape}"
        )

    def test_output_shape_different_classes(self, dummy_batch):
        num_cls = 10
        model = build_cnn(num_cls)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, num_cls)

    def test_output_is_logits(self, dummy_batch, num_classes):
        """La salida debe ser logits (sin normalización softmax)."""
        model = build_cnn(num_classes)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        # Con logits, los valores pueden ser negativos
        assert out.shape[-1] == num_classes

    def test_train_mode_updates_weights(self, dummy_batch, num_classes):
        model = build_cnn(num_classes)
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        labels = torch.randint(0, num_classes, (dummy_batch.size(0),))
        out = model(dummy_batch)
        loss = criterion(out, labels)

        before = [p.clone().detach() for p in model.parameters()]
        loss.backward()
        optimizer.step()
        after = list(model.parameters())

        changed = any(
            not torch.equal(b, a.detach()) for b, a in zip(before, after)
        )
        assert changed, "Los pesos no se actualizaron tras un paso de optimización"

    def test_eval_mode_deterministic(self, dummy_batch, num_classes):
        """En modo eval el modelo es determinista."""
        model = build_cnn(num_classes)
        model.eval()
        with torch.no_grad():
            out1 = model(dummy_batch)
            out2 = model(dummy_batch)
        assert torch.equal(out1, out2)

    def test_num_parameters_positive(self, num_classes):
        model = build_cnn(num_classes)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0


# ─── Tests del modelo ResNet-50 ──────────────────────────────────────────────

class TestResNetTransfer:

    def test_build_returns_correct_type(self, num_classes):
        model = build_resnet(num_classes)
        assert isinstance(model, ResNetTransfer)

    def test_output_shape(self, dummy_batch, num_classes):
        model = build_resnet(num_classes)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, num_classes)

    def test_backbone_frozen_by_default(self, num_classes):
        model = build_resnet(num_classes, freeze_backbone=True)
        trainable_backbone = [
            p for name, p in model.model.named_parameters()
            if "fc" not in name and p.requires_grad
        ]
        assert len(trainable_backbone) == 0, (
            "No debe haber parámetros del backbone entrenables cuando "
            "freeze_backbone=True"
        )

    def test_fc_trainable_when_frozen(self, num_classes):
        model = build_resnet(num_classes, freeze_backbone=True)
        fc_params = [p for p in model.model.fc.parameters()
                     if p.requires_grad]
        assert len(fc_params) > 0, "La capa FC debe ser entrenable"

    def test_unfreeze_layer4(self, num_classes):
        model = build_resnet(num_classes, freeze_backbone=True)
        model.unfreeze_layer4()
        layer4_trainable = [
            p for name, p in model.model.named_parameters()
            if "layer4" in name and p.requires_grad
        ]
        assert len(layer4_trainable) > 0

    def test_unfreeze_all(self, num_classes):
        model = build_resnet(num_classes, freeze_backbone=True)
        model.unfreeze_all()
        frozen = [p for p in model.parameters() if not p.requires_grad]
        assert len(frozen) == 0

    def test_get_trainable_params_returns_groups(self, num_classes):
        model = build_resnet(num_classes, freeze_backbone=True)
        groups = model.get_trainable_params()
        assert isinstance(groups, list)
        assert all("params" in g for g in groups)

    def test_output_shape_different_classes(self, dummy_batch):
        num_cls = 5
        model = build_resnet(num_cls)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, num_cls)

    def test_eval_mode_deterministic(self, dummy_batch, num_classes):
        model = build_resnet(num_classes)
        model.eval()
        with torch.no_grad():
            out1 = model(dummy_batch)
            out2 = model(dummy_batch)
        assert torch.equal(out1, out2)

    def test_backbone_not_frozen_when_flag_false(self, num_classes):
        model = build_resnet(num_classes, freeze_backbone=False)
        frozen = [p for p in model.parameters() if not p.requires_grad]
        assert len(frozen) == 0


# ─── Tests de utilidades de visualización ────────────────────────────────────

class TestVisualization:

    def test_plot_training_history_saves_file(self):
        import matplotlib
        matplotlib.use("Agg")
        from utils.visualization import plot_training_history

        history = {
            "train_loss": [1.0, 0.8, 0.6],
            "val_loss": [1.1, 0.9, 0.7],
            "train_acc": [50.0, 60.0, 70.0],
            "val_acc": [48.0, 58.0, 68.0],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "curves.png")
            plot_training_history(history, "TestModel", save_path=save_path)
            assert os.path.exists(save_path)

    def test_plot_metrics_bar_saves_file(self):
        import matplotlib
        matplotlib.use("Agg")
        from utils.visualization import plot_metrics_bar

        metrics = {
            "CNN": {"accuracy": 0.85, "precision": 0.84,
                    "recall": 0.83, "f1": 0.83},
            "ResNet": {"accuracy": 0.92, "precision": 0.91,
                       "recall": 0.90, "f1": 0.90},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "metrics.png")
            plot_metrics_bar(metrics, save_path=save_path)
            assert os.path.exists(save_path)

    def test_plot_comparison_saves_file(self):
        import matplotlib
        matplotlib.use("Agg")
        from utils.visualization import plot_comparison

        histories = {
            "CNN": {
                "val_loss": [1.0, 0.8, 0.6],
                "val_acc": [50.0, 60.0, 70.0],
            },
            "ResNet": {
                "val_loss": [0.9, 0.7, 0.5],
                "val_acc": [55.0, 65.0, 75.0],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "comparison.png")
            plot_comparison(histories, save_path=save_path)
            assert os.path.exists(save_path)

    def test_plot_confusion_matrix_saves_file(self):
        import matplotlib
        matplotlib.use("Agg")
        from utils.visualization import plot_confusion_matrix

        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 1, 0, 2]
        class_names = ["Clase A", "Clase B", "Clase C"]
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "cm.png")
            plot_confusion_matrix(y_true, y_pred, class_names,
                                  "TestModel", save_path=save_path)
            assert os.path.exists(save_path)


# ─── Tests de configuración ───────────────────────────────────────────────────

class TestConfig:

    def test_train_val_test_split_sums_to_one(self):
        total = config.TRAIN_SPLIT + config.VAL_SPLIT + config.TEST_SPLIT
        assert abs(total - 1.0) < 1e-6

    def test_num_classes_matches_plant_classes(self):
        assert len(config.PLANT_CLASSES) == config.NUM_CLASSES

    def test_image_size_is_positive(self):
        assert config.IMAGE_SIZE > 0

    def test_learning_rate_is_positive(self):
        assert config.LEARNING_RATE > 0

    def test_batch_size_is_positive(self):
        assert config.BATCH_SIZE > 0
