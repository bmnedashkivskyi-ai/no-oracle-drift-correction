# Схема системи autoresearch для SAM2

## Замкнений цикл

```mermaid
flowchart LR
    A[program.md: правила і бюджет] --> B[LLM agent]
    B --> C[Гіпотеза]
    C --> D[Зміна wrapper / prompt policy / fine-tuning code]
    D --> E[Git commit]
    E --> F[SAM2 inference або fine-tuning]
    F --> G[Маски і треки]
    G --> H[Ground truth evaluator]
    H --> I[J&F, IoU, boundary F1, latency, VRAM]
    I --> J{Покращення?}
    J -->|так| K[Keep + results.tsv]
    J -->|ні| L[Discard / revert]
    K --> B
    L --> B
```

## Шари пошуку

```mermaid
flowchart TB
    S[SAM2 checkpoint]
    S --> P[Prompt policy]
    S --> I[Inference settings]
    S --> M[Memory / propagation]
    S --> R[Post-processing]
    S --> T[Fine-tuning]
    P --> E[Evaluator]
    I --> E
    M --> E
    R --> E
    T --> E
    E --> D[Decision gate]
```

## Рекомендований порядок

1. Prompt policy: grid/random click selection, positive/negative points, box expansion, iterative correction.
2. Mask selection: score calibration, stability score, area and connected-component filters.
3. Video propagation: keyframe cadence, re-prompt on confidence drop, memory eviction.
4. Efficiency: BF16/FP16, image encoder compilation, batching, resolution and caching.
5. Fine-tuning: лише після того, як benchmark і data split стабільні.

## Контрольні точки

- До запуску: frozen checkpoint, dataset manifest і метрики.
- Після кожного кандидата: raw predictions, resources і commit.
- Перед прийняттям: held-out test і повтори.
- Після серії: error taxonomy, ablation і Pareto frontier.
