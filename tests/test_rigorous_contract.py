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
from cvlab.tuning.rigorous import RunSeeds, SingleRunResult, plan_run_seeds, rigorous_compare


class FakeDataModule:
    in_channels = 1
    num_classes = 2
    class_names = ["a", "b"]
    uses_official_split = False

    def configure_run_seeds(self, **_: Any) -> None: ...
    def setup(self, *_: Any) -> None: ...


def make_cfg(n_seeds: int = 3, objective: str = "accuracy", **variance: bool) -> Any:
    return OmegaConf.create(
        {
            "seed": 42,
            "tuning": {"n_seeds": n_seeds, "final_epochs": 1, "objective": objective},
            "trainer": {"accelerator": "cpu"},
            "variance": {"vary_init": True, "vary_data_order": False, "vary_split": False, **variance},
        }
    )


def fake_runs(
    monkeypatch: pytest.MonkeyPatch,
    accs: dict[str, list[tuple[float, float]]],
    scores: dict[str, list[float]] | None = None,
) -> list[str]:
    """Substitui o treino por acurácias fixas: (val_acc, test_acc) por seed e por arm.

    ``scores`` permite dissociar a métrica de SELEÇÃO da de relatório; sem ele,
    o score de validação é a própria ``val_acc`` (o caso ``objective=accuracy``).
    """
    calls = {"baseline": 0, "tuned": 0}
    seen_objectives: list[str] = []

    def fake_train(config, seeds, datamodule, epochs, accelerator="auto", pl_logger=None, objective="accuracy"):
        arm = "baseline" if config.get("arm") == "baseline" else "tuned"
        i = calls[arm]
        val, test = accs[arm][i]
        score = scores[arm][i] if scores is not None else val
        calls[arm] += 1
        seen_objectives.append(objective)
        n = 4
        per_split = {
            "val": {"preds": [0] * n, "targets": [0] * n, "probs": [[0.6, 0.4]] * n},
            "test": {
                "preds": [0, 1, 0, 1],
                "targets": [0, 0, 1, 1],
                "probs": [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]],
            },
        }
        epoch_rows = [
            {"epoch": 0, "train_loss": 0.9, "train_acc": 0.5, "val_loss": 0.8, "val_acc": val, "lr": 1e-3},
            {"epoch": 1, "train_loss": 0.5, "train_acc": 0.7, "val_loss": 0.6, "val_acc": val, "lr": 1e-3},
        ]
        return SingleRunResult(
            val_acc=val,
            test_acc=test,
            val_score=score,
            model=fake_model(),
            per_split=per_split,
            epoch_rows=epoch_rows,
        )

    monkeypatch.setattr(rigorous, "_train_single_run", fake_train)
    return seen_objectives


def fake_model() -> Any:
    """Um objeto qualquer no lugar do modelo: `rigorous_compare` só o repassa.

    Só a identidade importa — os testes checam QUAL modelo é selecionado, nunca
    o que ele faz. É o que permite exercitar a regra de decisão sem treinar.
    """
    return object()


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

    # probabilidades acompanham as predições: sem elas não há AUC nem calibração
    assert all(len(p["probs"]) == len(p["preds"]) for p in res["predictions"])

    # uma linha por (arm, seed, época) para as curvas de aprendizado
    assert len(res["epoch_rows"]) == 3 * 2 * 2
    assert {r["arm"] for r in res["epoch_rows"]} == {"baseline", "tuned"}
    assert {r["epoch"] for r in res["epoch_rows"]} == {0, 1}


def test_selection_uses_objective_score_not_reported_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Com `balanced_accuracy`, quem decide é o score — não a acurácia reportada.

    A tunada tem acurácia simples PIOR e acurácia balanceada MELHOR. Num dataset
    desbalanceado é exatamente esse o trade-off: acertar as classes raras custa
    acertos na majoritária. Se a seleção olhasse a coluna `acc`, a config que
    trata o desbalanceamento seria descartada justamente onde ela importa.
    """
    fake_runs(
        monkeypatch,
        accs={"baseline": [(0.90, 0.90)] * 3, "tuned": [(0.70, 0.70)] * 3},
        scores={"baseline": [0.50] * 3, "tuned": [0.65] * 3},
    )
    res = rigorous_compare(BASELINE, TUNED, FakeDataModule(), make_cfg(objective="balanced_accuracy"))

    assert res["tuned_ge_baseline"] is True
    assert res["selected_arm"] == "tuned"
    # A acurácia REPORTADA continua sendo a simples, e continua menor.
    assert res["best_val_mean"] < res["baseline_val_mean"]
    # E o manifest carrega o critério, senão o relatório não explica o acima.
    assert res["objective"] == "balanced_accuracy"
    assert res["best_val_score_mean"] == pytest.approx(0.65)


def test_objective_reaches_the_training_function(monkeypatch: pytest.MonkeyPatch) -> None:
    """`tuning.objective` chega a `_train_single_run`, que escolhe a época por ele.

    Sem isso o early stopping otimizaria uma métrica e a seleção final, outra.
    """
    seen = fake_runs(monkeypatch, {"baseline": [(0.5, 0.5)] * 2, "tuned": [(0.6, 0.6)] * 2})
    rigorous_compare(BASELINE, TUNED, FakeDataModule(), make_cfg(n_seeds=2, objective="balanced_accuracy"))
    assert seen == ["balanced_accuracy"] * 4


def test_objective_defaults_to_accuracy_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config sem `objective` reproduz o protocolo histórico, sem quebrar."""
    cfg = make_cfg(n_seeds=2)
    del cfg.tuning.objective
    seen = fake_runs(monkeypatch, {"baseline": [(0.5, 0.5)] * 2, "tuned": [(0.6, 0.6)] * 2})
    res = rigorous_compare(BASELINE, TUNED, FakeDataModule(), cfg)
    assert seen == ["accuracy"] * 4
    assert res["objective"] == "accuracy"


def test_default_varies_only_initialization() -> None:
    """O default reproduz o protocolo histórico: só os pesos variam entre seeds."""
    seeds = plan_run_seeds(make_cfg(n_seeds=3), 3)
    assert seeds == [RunSeeds(42, 42, 42), RunSeeds(43, 42, 42), RunSeeds(44, 42, 42)]


def test_enabling_all_sources_varies_all_three() -> None:
    """Ligar as três fontes mede a variância que o protocolo default esconde."""
    cfg = make_cfg(n_seeds=2, vary_data_order=True, vary_split=True)
    assert plan_run_seeds(cfg, 2) == [RunSeeds(42, 42, 42), RunSeeds(43, 43, 43)]
