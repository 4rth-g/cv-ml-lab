# cv-ml-lab

Framework para classificação de imagens com PyTorch Lightning, Hydra, Optuna e
Weights & Biases. Aplica o mesmo procedimento de busca de hiperparâmetros e
comparação estatística a qualquer dataset de imagem, definido por configuração em
vez de código.

## Navegação

- [Fluxo de trabalho](WORKFLOW.md) — exploração em notebooks e formalização neste framework.
- [Referência da API](reference/tuning.md) — gerada a partir das docstrings do pacote.

## Método, em resumo

A busca e a comparação são etapas separadas, e cada passo existe para remover um
viés específico:

1. **Busca (Optuna)** em um subconjunto do treino, com a baseline enfileirada como
   primeiro trial — o melhor da busca nunca fica abaixo da baseline.
2. **Retreino multi-seed** de baseline e configuração tunada no split completo,
   restaurando os pesos da melhor época segundo a validação.
3. **Comparação pareada**: teste t pareado, Cohen's d_z, IC95% e McNemar — as mesmas
   seeds nos dois lados tornam as amostras pareadas e cancelam o ruído seed-a-seed.
4. **Seleção na validação**, com *fallback* para a baseline se a tunada não a
   superar. O conjunto de teste é usado uma única vez, apenas para reportar.

O detalhamento de cada passo está nas docstrings de
[`cvlab.tuning.search`](reference/tuning.md) e
[`cvlab.tuning.rigorous`](reference/tuning.md).
