# Modelos

Fábrica de modelos (instanciação por configuração), arquiteturas e o wrapper
`LitClassifier` comum a todas.

## Fábrica de modelos

::: cvlab.models.factory

## Wrapper de treino

Otimizador e scheduler são escolhidos por nome, a partir das tabelas
`OPTIMIZERS` e `SCHEDULERS`. Isso os torna hiperparâmetros de busca como
quaisquer outros — se o cosine ajuda numa arquitetura é o método que responde,
não uma escolha embutida no código.

::: cvlab.models.lit_module

## Arquiteturas

::: cvlab.models.cnn

::: cvlab.models.mlp

::: cvlab.models.perceptron

::: cvlab.models.resnet
