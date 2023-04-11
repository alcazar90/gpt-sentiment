# GPT sentiment

Estrategia:

1. Entrenar un modelo no-supervisado (autoregresivo) para generación de texto
2. Utilizar el modelo para crear una representación de los datos (i.e. embeddings)
3. Agregar una cabeza de clasificación y utilizar la representación sobre el texto del input y las clases


Se puede ver las configuraciones de entrenamiento en W&B: [`sentiment_gpt_esp`](https://wandb.ai/alcazar90/sentiment_gpt_esp?workspace=user-alcazar90).

## acknowledgements

El código fue adaptado del proyecto [nanoGPT](https://github.com/karpathy/nanoGPT).

