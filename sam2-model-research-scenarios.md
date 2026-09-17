# Сценарії покращення самої SAM2-моделі

Дата: 2026-08-28

## Межа між prompt policy і model-level research

Попередня серія покращила спосіб подачі інформації до frozen SAM2.1 Hiera-Tiny. Щоб покращити саму модель, кандидат повинен змінювати параметри, навчання, дані, loss або модельну архітектуру і проходити контрольний retraining/fine-tuning.

Кожен сценарій потребує:

- frozen held-out test;
- однаковий training budget;
- однаковий checkpoint initialization;
- versioned dataset manifest;
- щонайменше 3 seeds для значущого кандидата;
- перевірки на DAVIS і незалежному SA-V validation.

## Сценарій A: domain fine-tuning на DAVIS

**Гіпотеза:** коротке fine-tuning decoder/memory на DAVIS покращить region і boundary quality.

**Зміни:**

- спочатку freeze image encoder;
- fine-tune mask decoder і memory attention;
- перевірити full fine-tuning як окремий кандидат;
- використовувати BCE/Dice/focal loss для masks.

**Ризик:** overfitting до DAVIS і втрата zero-shot generalization.

**Метрики:** DAVIS `J&F`, decay, SA-V `J&F`, cross-domain degradation.

## Сценарій B: fine-tuning на SA-V manual masklets

**Гіпотеза:** manual spatio-temporal masklets поліпшать складні рухи, occlusion і appearance changes.

**Зміни:** використовувати SA-V train з узгодженим frame sampling і manual annotations; auto masklets застосовувати лише для ablation.

**Ризик:** величезний dataset і значний compute budget; ліцензія CC BY 4.0 потребує attribution.

**Метрики:** SA-V val/test `J`, `F`, `J&F`, object-size buckets, visibility-change buckets.

## Сценарій C: навчання з hard-negative prompts

**Гіпотеза:** модель стане менш чутливою до prompt ambiguity, якщо training включатиме точки біля межі, поза об'єктом і на сусідніх об'єктах.

**Зміни:**

- генерувати positive points за erosion distance;
- negative points за dilation ring;
- додавати box noise 0%, 5%, 10%, 20%;
- оптимізувати loss одночасно на clean і noisy prompts.

**Метрики:** quality як функція prompt noise, `J&F` при однаковому prompt budget, failure rate.

## Сценарій D: temporal consistency loss

**Гіпотеза:** допоміжний loss між масками сусідніх кадрів зменшить flicker і identity failures.

**Зміни:**

- використовувати optical-flow або feature correspondence;
- штрафувати непояснені зміни маски;
- не штрафувати реальні visibility changes з annotations.

**Ризик:** oversmoothing меж і погіршення швидких рухів.

**Метрики:** `J`, `F`, temporal IoU, flicker rate, identity preservation.

## Сценарій E: memory policy і memory encoder

**Гіпотеза:** адаптивний відбір memory tokens буде кращим за фіксовану історію.

**Зміни:**

- trainable memory eviction score;
- diverse keyframe selection;
- окремі memory banks для appearance і motion;
- confidence-triggered memory refresh.

**Ризик:** нестабільність на довгих відео й зростання latency.

**Метрики:** long-video `J&F`, memory size, FPS, peak VRAM, failure rate.

## Сценарій F: model distillation для Hiera-Tiny

**Гіпотеза:** Hiera-Large/Small teacher може передати Tiny кращі boundary features при меншій inference cost.

**Зміни:**

- distillation logits/masks/features;
- boundary-aware distillation loss;
- перевірити teacher confidence filtering.

**Метрики:** якість Tiny проти baseline, latency, VRAM і model size.

## Сценарій G: backbone/decoder capacity search

**Гіпотеза:** обмежений autoresearch по depth, width, decoder heads і attention channels знайде кращу Pareto-точку.

**Зміни:**

- Hiera Tiny width/depth;
- mask decoder depth;
- memory attention heads;
- parameter sharing між frames.

**Правило:** кожен candidate має мати однаковий або явно зафіксований FLOPs budget.

**Метрики:** `J&F` при фіксованих FLOPs, latency, VRAM, parameters.

## Сценарій H: data engine для active learning

**Гіпотеза:** агент може вибирати кадри, де model uncertainty і boundary error максимальні, замість рівномірного annotation budget.

**Зміни:**

- uncertainty sampling;
- diversity sampling між відео;
- human correction simulation;
- повторне навчання на відібраних masklets.

**Метрики:** `J&F` на annotation budget, quality per labeled frame, domain transfer.

## Сценарій I: loss architecture search

**Гіпотеза:** комбінація region, boundary, IoU і temporal losses буде кращою за один mask loss.

**Кандидати:**

- BCE + Dice;
- focal + Dice;
- Lovasz-IoU;
- boundary distance transform;
- temporal consistency auxiliary loss.

**Ризик:** loss tuning може оптимізувати benchmark без реального покращення візуальної якості.

## Сценарій J: system-aware joint search

**Гіпотеза:** найкраща production-модель є не найточнішою, а Pareto-optimal за `J&F`, FPS, VRAM і interaction count.

**Зміни:** спільно шукати checkpoint training recipe, model size і memory policy.

**Ціль:** не один scalar score, а Pareto frontier; кожне рішення перевіряти на held-out split.

## Пріоритет

1. C: hard-negative prompts — найдешевший model-level fine-tuning.
2. D: temporal consistency loss — найбільш релевантний для video propagation.
3. F: distillation — підходить для RTX 4070 Ti та Tiny deployment.
4. E: memory encoder/policy — потенційно великий ефект на довгих відео.
5. A/B: domain fine-tuning — після стабілізації benchmark.
6. G/H/I/J: повний research program із більшим budget.

## Acceptance criteria

Model-level candidate приймається, якщо:

- покращує `J&F` мінімум на 1 pp на DAVIS held-out test;
- не втрачає більше 0.5 pp на SA-V transfer test;
- результат відтворюється на 3 seeds;
- немає критичної деградації на малих об'єктах або visibility changes;
- resource trade-off задокументований;
- є ablation, яка показує внесок кожної зміни.

## Джерела

- SAM2 paper: https://arxiv.org/abs/2408.00714
- SAM2 official code and training: https://github.com/facebookresearch/sam2
- SA-V dataset/evaluator: https://github.com/facebookresearch/sam2/tree/main/sav_dataset
- DAVIS evaluator: https://github.com/davisvideochallenge/davis2017-evaluation
- autoresearch protocol: https://github.com/karpathy/autoresearch
- Modded-NanoGPT methodology: https://github.com/KellerJordan/modded-nanogpt
