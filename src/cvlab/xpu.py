"""Accelerator Intel XPU (Arc) custom para o Lightning.

O Lightning 2.x reconhece apenas cpu/cuda/mps/tpu/hpu nativamente — Intel Arc
(XPU) não é first-class. Este módulo registra um accelerator ``"xpu"`` para
treinar em placas Intel Arc com builds PyTorch ``+xpu``. Importar este módulo já
registra o accelerator no ``AcceleratorRegistry``.
"""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch.accelerators.accelerator import Accelerator

try:
    from lightning.pytorch.accelerators import AcceleratorRegistry
except Exception:  # pragma: no cover - variações de versão do Lightning
    AcceleratorRegistry = None


class XPUAccelerator(Accelerator):
    """Accelerator para GPUs Intel (XPU / Arc), single-device."""

    def setup_device(self, device: torch.device) -> None:
        if device.type != "xpu":
            raise ValueError(f"Device esperado do tipo 'xpu', recebido: {device}")
        torch.xpu.set_device(device)

    def teardown(self) -> None:
        torch.xpu.empty_cache()

    @staticmethod
    def parse_devices(devices: Any) -> list[int]:
        if isinstance(devices, int):
            return list(range(devices))
        if isinstance(devices, str):
            if devices == "auto":
                return list(range(XPUAccelerator.auto_device_count()))
            return [int(d) for d in devices.split(",") if d.strip() != ""]
        return list(devices)

    @staticmethod
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        return [torch.device("xpu", i) for i in devices]

    @staticmethod
    def auto_device_count() -> int:
        return torch.xpu.device_count() if XPUAccelerator.is_available() else 0

    @staticmethod
    def is_available() -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    @staticmethod
    def name() -> str:
        return "xpu"

    def get_device_stats(self, device: str | torch.device) -> dict[str, Any]:
        return {}


if AcceleratorRegistry is not None:
    try:
        AcceleratorRegistry.register("xpu", XPUAccelerator, description="Intel XPU (Arc) accelerator")
    except Exception:  # já registrado (reimport) ou versão incompatível
        pass


def build_trainer(accelerator: str, **kwargs: Any):
    """Constrói um lightning.Trainer, forçando o device XPU quando aplicável.

    O connector do Lightning não propaga o device XPU para a estratégia (cai em
    cpu), então no modo 'xpu' passamos uma SingleDeviceStrategy explícita em
    xpu:0. Para os demais accelerators ('auto'/'cpu'/'gpu'...) delega ao Lightning.
    """
    import lightning as L

    if accelerator == "xpu" and XPUAccelerator.is_available():
        from lightning.pytorch.strategies import SingleDeviceStrategy

        strategy = SingleDeviceStrategy(device=torch.device("xpu", 0), accelerator=XPUAccelerator())
        return L.Trainer(strategy=strategy, devices=1, **kwargs)
    return L.Trainer(accelerator=accelerator, devices=1, **kwargs)
