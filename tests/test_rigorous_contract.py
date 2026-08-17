"""Testes da regra de DECISÃO do `rigorous_compare`, sem treinar nada.

Com a inferência estatística migrando para o R, o que resta de estatística no
Python é exatamente isto: a seleção baseline-vs-tunada e a escolha da seed que
alimenta o McNemar. São as duas coisas que mudam o modelo entregue, então são as
duas que precisam de trava de regressão.
"""

from typing import Any

import pytest
from omegaconf import OmegaConf

from cvlab.tuning import rigorous
from cvlab.tuning.rigorous import RunSeeds, plan_run_seeds, rigorous_compare


class FakeDataModule:
    in_channels = 1
    num_classes = 2
    class_names = ["a", "b"]
    uses_official_split = False

    def configure_run_seeds(self, **_: Any) -> None: ...
    def setup(self, *_: Any) -> None: ...


def make_cfg(n_seeds: int = 3, **variance: bool) -> Any:
    return OmegaConf.create(
        {
            "seed": 42,
            "tuning": {"n_seeds": n_seeds, "final_epochs": 1},
            "trainer": {"accelerator": "cpu"},
            "variance": {"vary_init": True, "vary_data_order": False, "vary_split": False, **variance},
        }
    )


def fake_runs(monkeypatch: pytest.MonkeyPatch, accs: dict[str, list[tuple[float, float]]]) -> None:
    """Substitui o treino por acurácias fixas: (val_acc, test_acc) por seed e por arm."""
    calls = {"baseline": 0, "tuned": 0}

    def fake_train(config, seeds, datamodule, epochs, accelerator="auto", pl_logger=None):
        arm = "baseline" if config.get("arm") == "baseline" else "tuned"
        val, test = accs[arm][calls[arm]]
        calls[arm] += 1
        n = 4
        per_split = {
            "val": ([0] * n, [0] * n),
            "test": ([0, 1, 0, 1], [0, 0, 1, 1]),
        }
        return test, val, object(), per_split

    monkeypatch.setattr(rigorous, "_train_single_run", fake_train)


BASELINE = {"_target_": "x", "arm": "baseline"}
TUNED = {"_target_": "x", "arm": "tuned"}


def test_selection_uses_validation_not_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tunada perde na validação e VENCE no teste: deve haver fallback para a baseline.

    É o guard-rail central do método. Se algum dia a seleção olhar o teste, este
    teste falha — e é a diferença entre reportar um resultado e inventá-lo.
    """
    fake_runs(
        monkeypatch,
        {
            "baseline": [(0.90, 0.10), (0.90, 0.10), (0.90, 0.10)],
            "tuned": [(0.50, 0.99), (0.50, 0.99), (0.50, 0.99)],
        },
    )
    res = rigorous_compare(BASELINE, TUNED, FakeDataModule(), make_cfg())

    assert res["tuned_ge_baseline"] is False
    assert res["selected_arm"] == "baseline"
    assert res["selected_config"] is BASELINE


def test_tie_on_validation_goes_to_tuned(monkeypatch: pytest.MonkeyPatch) -> None:
    """A regra é `>=`: empate na validação escolhe a tunada."""
    fake_runs(
        monkeypatch,
        {"baseline": [(0.80, 0.5)] * 3, "tuned": [(0.80, 0.7)] * 3},
    )
    res = rigorous_compare(BASELINE, TUNED, FakeDataModule(), make_cfg())
    assert res["tuned_ge_baseline"] is True
    assert res["selected_arm"] == "tuned"


def test_mcnemar_seed_is_best_validation_never_best_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """A seed do McNemar sai da melhor VALIDAÇÃO. Usar a de melhor teste seria vazamento."""
    fake_runs(
        monkeypatch,
        {
            # seed 43 é a melhor em validação; seed 44 é a melhor em teste.
            "baseline": [(0.10, 0.10), (0.90, 0.20), (0.20, 0.99)],
            "tuned": [(0.10, 0.10), (0.95, 0.20), (0.20, 0.99)],
        },
    )
    res = rigorous_compare(BASELINE, TUNED, FakeDataModule(), make_cfg())
    assert res["mcnemar_baseline_seed"] == 43
    assert res["mcnemar_tuned_seed"] == 43


def test_export_payload_is_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """O retorno carrega o que o export precisa, em formato longo."""
    fake_runs(monkeypatch, {"baseline": [(0.5, 0.5)] * 3, "tuned": [(0.6, 0.6)] * 3})
    res = rigorous_compare(BASELINE, TUNED, FakeDataModule(), make_cfg())

    assert len(res["seed_rows"]) == 3 * 2 * 2
    assert {r["split"] for r in res["seed_rows"]} == {"val", "test"}
    assert {r["arm"] for r in res["seed_rows"]} == {"baseline", "tuned"}
    # predições dos DOIS splits, para que a métrica possa ser escolhida depois
    assert len(res["predictions"]) == 3 * 2 * 2
    assert {p["split"] for p in res["predictions"]} == {"val", "test"}


def test_default_varies_only_initialization() -> None:
    """O default reproduz o protocolo histórico: só os pesos variam entre seeds."""
    seeds = plan_run_seeds(make_cfg(n_seeds=3), 3)
    assert seeds == [RunSeeds(42, 42, 42), RunSeeds(43, 42, 42), RunSeeds(44, 42, 42)]


def test_enabling_all_sources_varies_all_three() -> None:
    """Ligar as três fontes mede a variância que o protocolo default esconde."""
    cfg = make_cfg(n_seeds=2, vary_data_order=True, vary_split=True)
    assert plan_run_seeds(cfg, 2) == [RunSeeds(42, 42, 42), RunSeeds(43, 43, 43)]
