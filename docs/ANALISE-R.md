# Análise estatística (R)

O cvlab é um pipeline de **duas linguagens**, com uma divisão que não é de gosto
e sim de responsabilidade:

- **Python treina, decide e exporta.** Reduz imagens e modelos a tabelas.
- **R infere e reporta.** Lê essas tabelas e produz testes, correções, intervalos
  e gráficos.

A regra que sustenta a separação:

> O R nunca toca em uma imagem. O `analysis/` depende **apenas** do contrato de
> arquivos, nunca de código Python. Sem `reticulate`, sem importar de
> `src/cvlab/`, sem caminho absoluto.

## Por que separar

Porque a inferência nunca deveria ter influência sobre o modelo entregue. A
seleção baseline-vs-tunada acontece em
[`rigorous_compare`](reference/tuning.md), na **validação**, e usa apenas médias.
Como nenhum p-valor é calculado durante o treino, nenhum p-valor pode vazar para
a decisão. Isso deixou de ser uma convenção de disciplina e passou a ser uma
propriedade estrutural do código.

O ganho prático é maior ainda: como as **predições por exemplo** são exportadas,
qualquer métrica vira cálculo pós-hoc. F1 macro, balanced accuracy, recall por
classe, AUC, kappa quadrático e matriz de confusão saem de runs **já feitos**,
sem retreinar e sem tocar no `LitClassifier`. A escolha da métrica deixou de ser
decisão de engenharia e virou decisão de análise.

## O contrato de arquivos

```
results/runs/<run_id>/manifest.json        metadados, configs, seeds do McNemar
results/runs/<run_id>/seed_metrics.csv     uma linha por (arm, seed, split)
results/runs/<run_id>/predictions.csv.gz   uma linha por (arm, seed, split, exemplo)
results/runs/<run_id>/trials.csv           histórico da busca Optuna
results/runs/index.csv                     uma linha por run (append-only)

results/datasets/<dataset_id>/summary.json    agregados por split e classe
results/datasets/<dataset_id>/examples.csv.gz uma linha por exemplo
```

A chave `(split, example_idx)` liga os dois últimos aos primeiros. É ela que
permite cruzar **erro do modelo** com **propriedade da imagem**, coisa que
notebooks separados nunca deram.

Cada artefato carrega `schema_version`. O leitor R **recusa** o que não
reconhece, em vez de interpretar errado em silêncio.

## Uso

```bash
just report            # análise do run mais recente (texto)
just report <run_id>   # de um run específico
just report-all        # agregado: p corrigido, Friedman + Wilcoxon-Holm
just report-html       # relatório HTML autocontido -> analysis/output/report.html
just report-preview    # servidor com recarga automática a cada save
just report-test       # suíte testthat de analysis/
```

### Visualizando o relatório

`just report-html` grava `analysis/output/report.html`, um arquivo autocontido
(imagens embutidas em base64) que abre direto no navegador:

```bash
just report-html latest results/_smoke
xdg-open analysis/output/report.html
```

Para editar o `.qmd` com recarga automática, `just report-preview` sobe um
servidor local e re-renderiza a cada save.

### Ferramentas fora do justfile (`quarto preview`, VS Code, RStudio)

As recipes entram no devShell sozinhas, mas ferramentas que **não** passam pelo
`just` — a extensão Quarto do VS Code, um `quarto preview` digitado à mão, o
RStudio — rodam no ambiente do editor e não encontram o R.

A solução é o `direnv`, que já vem configurado no `.envrc` do repositório:

```bash
direnv allow    # uma vez, no clone
```

A partir daí o devShell é ativado ao entrar no diretório, e qualquer ferramenta
lançada dali enxerga o R e os pacotes. Sem isso, o Quarto falha com
`Unable to locate an installed version of R`.

Todas aceitam um segundo argumento com a raiz de resultados, **relativa à raiz do
repositório**, para analisar artefatos que não estão em `results/`:

```bash
just report-html latest results/_smoke
```

As recipes checam antes se existe `<raiz>/runs/index.csv` e, se não existir,
falham com instruções em vez de deixar o erro aparecer no meio do backtrace do
Quarto. A análise **lê** artefatos, não treina: é preciso rodar `just train` (ou
`just smoke`) primeiro.

Não é preciso entrar no `nix develop` antes: o R deste projeto vive só no flake,
então as recipes detectam sua ausência no PATH e entram no devShell sozinhas.
Rodar de dentro de `nix develop` também funciona, sem aninhar. `just r-shell`
abre o shell quando você quiser trabalhar interativamente em R.

O Quarto **não** vem do flake: o pacote do nixpkgs (1.10.18) empacota um pandoc
mais antigo que as opções que ele próprio gera, e `quarto render` falha com
`Unknown option "syntax-highlighting"`. O shell usa o Quarto instalado no
sistema e avisa se ele estiver ausente.

O EDA de dataset é gerado à parte, e só precisa rodar uma vez por dataset:

```bash
just eda medmnist dataset.subset=dermamnist
```

## O que o R faz que o Python não fazia

| | |
|---|---|
| **Correção para múltiplas comparações** | `p.adjust` (Holm e BH). Com dezenas de combinações dataset × arquitetura, algum p < 0,05 aparece por acaso quase sempre. Era a dívida mais grave do repo. |
| **IC do tamanho de efeito** | `effectsize::cohens_d(paired = TRUE)`. O Python dava o d_z como número solto, sem incerteza. |
| **Estimativas robustas** | IQM e bootstrap BCa. Com 5 seeds, a média é dominada por outlier e a mediana joga informação fora. |
| **P(tunada > baseline)** | Responde "vale trocar?", que o p-valor não responde. |
| **Bootstrap por exemplo** | Reamostra `example_idx`, separando a incerteza do test set finito da estocasticidade do treino. Só possível porque as predições são exportadas. |
| **Equivalência (TOST)** | Com poucas seeds, "não significativo" é o resultado mais provável e não significa "equivalente". |
| **Friedman + Wilcoxon-Holm** | Comparação entre arquiteturas ao longo de vários datasets. |
| **Métricas pós-hoc** | F1, balanced accuracy, kappa, confusão, retroativo sobre todos os runs. |
| **Erro cruzado com o EDA** | Erro por classe, correlação com intensidade, e quais exemplos **todas** as seeds erram. |

### Nemenyi não, Wilcoxon-Holm sim

O arcabouço é o de Demšar (JMLR 2006): Friedman com datasets como blocos. Mas o
post-hoc **não** é o Nemenyi que aquele artigo popularizou. Benavoli et al. (JMLR
2016) mostrou que testes de rank médio são não-transitivos e dependem de quais
outros algoritmos estão na comparação, a ponto de incluir um método ruim mudar a
conclusão sobre dois bons. O padrão atual é Wilcoxon pareado com correção de
Holm, que é o que `analysis/R/ranks.R` implementa.

## Fontes de variância

`seed_metrics.csv` tem **três** colunas de seed (`seed_init`, `seed_data`,
`seed_split`), e não uma. O motivo é concreto: variar só a inicialização dos
pesos mantém split e ordem de dados fixos e **subestima** a variância real, o que
produz intervalo estreito demais (Bouthillier et al., MLSys 2021).

O default do cvlab varia apenas `seed_init`, preservando o protocolo histórico.
As outras se ligam em `configs/train.yaml`:

```yaml
variance:
  vary_init: true
  vary_data_order: false
  vary_split: false
```

Ligá-las **alarga** o intervalo. Isso é correção, não piora. Independentemente da
escolha, as três colunas vão para o export, e o relatório sempre declara qual
variância foi medida.

## Paridade com a implementação anterior

Teste t, Cohen's d_z, IC95% e McNemar viviam em `rigorous.py` e `tracking.py`. Ao
migrar, a equivalência numérica foi verificada contra o `scipy`: as diferenças
ficaram em 1e-16 a 1e-18, ou seja epsilon de máquina, com `t`, `d_z` e o limite
superior do IC bit a bit idênticos.

Os casos que o antigo `tests/test_mcnemar.py` cobria viraram
`analysis/tests/testthat/test-parity.R`, com valores **literais** calculados à
mão, incluindo a fronteira de 25 discordâncias entre binomial exata e
qui-quadrado. Isso mantém a paridade verificável mesmo depois de o código Python
ter sido removido, sem que nenhum dos lados precise do outro em runtime.

## Adicionando uma análise

1. Escreva a função em `analysis/R/`, recebendo data frames e devolvendo data
   frame. Nada de I/O fora de `io.R`.
2. Exporte em `NAMESPACE`.
3. Teste em `analysis/tests/testthat/`, usando o fixture de
   `analysis/inst/extdata/results/` — que é um run real gerado pelo pipeline
   Python e funciona como contrato compartilhado entre as duas linguagens.
4. Chame de `run_report.R`.

Se um dia a análise precisar de algo que só existe em Python (`rliable`,
`autorank`, `baycomp`), o caminho é um script isolado em `analysis/py/` lendo os
**mesmos CSVs**. O contrato de arquivos já permite isso sem acoplar nada.
