# Fluxo de trabalho — exploração e formalização

O trabalho se organiza em dois níveis com papéis distintos: exploração em notebooks e formalização neste framework. O princípio: notebooks servem para explorar; o `cv-ml-lab` executa o experimento oficial, reprodutível.

## 1. Repositórios de exploração (notebooks)

Exemplos: `mnist-study`, `fashion-mnist-fundamentos-ia`, e futuros (um por dataset ou tema).

Propósito: entender os dados e prototipar rapidamente. O trabalho aqui pode ser iterativo e provisório.

- EDA: distribuições, amostras, normalização, classes difíceis.
- Protótipos: testar arquiteturas e configurações rapidamente (uma seed, sem rigor estatístico).
- Saída esperada: um candidato a formalizar, composto por dataset, arquitetura, faixas de hiperparâmetros promissoras e a augmentation apropriada ao domínio.
- Convenções: dados brutos fora do controle de versão (`data/` no `.gitignore`); reprodutibilidade total não é exigida nesta fase.

## 2. cv-ml-lab (formalização)

Onde o candidato validado se torna um experimento reprodutível, testado e registrado: CLI com Hydra (configuração versionada), método rigoroso (baseline no espaço de busca, retreino multi-seed, teste t pareado, Cohen's d_z e McNemar, seleção na validação), tracking no W&B e testes com CI.

## Pipeline de promoção

1. Explorar (repositório de notebooks): EDA e protótipo, produzindo um candidato.
2. Formalizar (cv-ml-lab): migrar dataset, arquitetura e espaço de busca; adicionar testes; executar via CLI na GPU (offline) e sincronizar com o W&B.
3. Reportar: os resultados ficam no W&B; a comparação estatística é o resultado a ser reportado.

## Checklist — adicionar um dataset

- [ ] `src/cvlab/data/<nome>.py`: `DataModule(BaseImageClfDataModule)` com `in_channels`, `num_classes`, `class_names`, `mean`, `std` e `_augment_ops()` apropriado ao domínio (por exemplo, dígitos não devem sofrer flip horizontal; roupas toleram flip).
- [ ] `configs/dataset/<nome>.yaml`.
- [ ] `tests/test_datamodule.py`: formatos e divisão treino/validação/teste.
- [ ] `cvlab-train dataset=<nome> ...` executa de ponta a ponta.

## Checklist — adicionar uma arquitetura

- [ ] `src/cvlab/models/<nome>.py`: `nn.Module` com `__init__(self, in_channels, num_classes, ...)` (parâmetros de arquitetura próprios) e `forward` retornando logits. Use `nn.LazyLinear` para não fixar a dimensão de entrada.
- [ ] `configs/model/<nome>.yaml` (baseline): `_target_` do modelo + valores de arquitetura e de treino (`lr`, `optimizer`, `weight_decay`, `batch_size`). Os valores baseline devem estar dentro do espaço de busca.
- [ ] `configs/tuning/<nome>.yaml`: `search_space` tipado — cada chave é `{type: categorical, choices: [...]}` ou `{type: float|int, low: x, high: y, log: bool}`. Parâmetros de treino (`lr`, `optimizer`, `weight_decay`, `batch_size`) são comuns; os demais vão ao construtor do modelo.
- [ ] `configs/experiment/<nome>.yaml`: preset `# @package _global_` que faz `override /model` e `override /tuning` para a arquitetura, permitindo `+experiment=<nome>`.
- [ ] `tests/test_<nome>.py`: formato do forward (in_channels 1 e 3; `num_classes` variável).
- [ ] `cvlab-train +experiment=<nome> dataset=<ds> ...` executa de ponta a ponta.

A fábrica de modelos (`models/factory.py`) e o objective genérico (`suggest_params`) dispensam qualquer código específico da arquitetura no tuning: basta os quatro arquivos de configuração acima.

## Execução e tracking

- Treino executado na GPU do desktop (Intel Arc B580).
- Quando o host de treino não alcança o W&B, registrar offline (`logger.mode=offline`) e sincronizar de outra máquina (`uvx wandb sync wandb/offline-run-*`).
- Runs nomeados `modelo-dataset-YYYY-MM-DD_HH-MM-SS` (ex.: `mlp-mnist-...`); projeto W&B `cv-ml-lab`.
- Estudo Optuna e tracker SQLite são artefatos de execução, ignorados pelo git e recriáveis.

## Controle de versão e publicação

- Um branch por feature; merge em `main` antes de publicar.
- `.gitignore` cobre `data/`, `.venv*/`, `wandb/`, `results/`, `*.db`, `*.pth`, `*.ckpt`, `.idea/`.
- Licença: MIT nos repositórios individuais; em trabalho em grupo, definir a licença com o acordo dos coautores antes de publicar.
- Segredos não entram no repositório (a chave do W&B fica apenas em `~/.netrc`).
