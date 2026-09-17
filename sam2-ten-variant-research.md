# Research: десять prompt-варіантів SAM2 на DAVIS

Дата: 2026-08-28

## Дизайн експерименту

Мета серії — перевірити, чи може autoresearch-підхід знаходити кращу стратегію взаємодії з frozen SAM2.1 Hiera-Tiny.

Фіксовані умови:

- checkpoint: SAM2.1 Hiera-Tiny;
- SAM2 upstream commit: `2b90b9f5ceec907a1c18123530e92e794ad901a4`;
- GPU: NVIDIA GeForce RTX 4070 Ti, 12 GB;
- PyTorch: `2.13.0+cu130`;
- dataset: DAVIS 2017 TrainVal 480p;
- test set: 10 validation sequences;
- budget: перші 20 кадрів кожної sequence;
- prompt: ground-truth-derived prompt на першому кадрі, що відповідає semi-supervised VOS protocol;
- evaluator: `scripts/davis_sam2_eval.py`;
- рішення: порівняння середніх `J`, `F`, `J&F`, latency, FPS і VRAM.

Десять sequence:

```text
bike-packing, blackswan, bmx-trees, camel, car-roundabout,
cows, dog, drift-chicane, goat, soapbox
```

Загальний обсяг: 100 оцінених кадрів на кожен варіант.

## Варіанти

| ID | Policy | Суть |
|---:|---|---|
| 1 | `centroid` | одна позитивна точка в centroid об'єкта |
| 2 | `box` | tight bounding box |
| 3 | `box_expand_05` | box із розширенням 5% |
| 4 | `box_expand_10` | box із розширенням 10% |
| 5 | `two_positive` | дві позитивні точки |
| 6 | `three_positive` | три позитивні точки |
| 7 | `box_centroid` | tight box + позитивна centroid point |
| 8 | `box_positive_negative` | tight box + positive centroid + negative point зовні |
| 9 | `box_expand_10_negative` | box із розширенням 10% + positive/negative points |
| 10 | `box_two_positive` | tight box + дві позитивні точки |

## Результати

Середні значення по 10 sequence:

| Policy | J | F | J&F | Latency ms | FPS | Peak VRAM MB |
|---|---:|---:|---:|---:|---:|---:|
| `centroid` | 0.8325 | 0.7931 | 0.8128 | 3392.5 | 6.36 | 733.4 |
| `box` | 0.8918 | 0.8568 | 0.8743 | 3389.3 | 6.35 | 733.4 |
| `box_expand_05` | 0.8898 | 0.8557 | 0.8727 | 3384.7 | 6.37 | 733.4 |
| `box_expand_10` | 0.8872 | 0.8565 | 0.8718 | 3399.7 | 6.33 | 733.4 |
| `two_positive` | 0.8901 | 0.8486 | 0.8694 | 3401.3 | 6.31 | 733.4 |
| `three_positive` | 0.8822 | 0.8458 | 0.8640 | 3409.1 | 6.27 | 733.4 |
| `box_centroid` | 0.9047 | 0.8648 | 0.8847 | 3424.7 | 6.30 | 733.4 |
| `box_positive_negative` | 0.9049 | 0.8645 | 0.8847 | 3399.3 | 6.36 | 733.4 |
| `box_expand_10_negative` | 0.8967 | 0.8600 | 0.8784 | 3432.1 | 6.27 | 733.4 |
| `box_two_positive` | 0.9016 | 0.8622 | 0.8819 | 3375.2 | 6.37 | 733.4 |

Raw CSV files знаходяться в `results/variants/`.

## Порівняння з baseline

У baseline `centroid`:

- `J&F = 0.8128`;
- `J = 0.8325`;
- `F = 0.7931`.

Найкращий за якістю `box_positive_negative`:

- `J&F = 0.8847`;
- приріст: `+0.0719` або `+7.19 percentage points`;
- `J = 0.9049`, приріст `+0.0724`;
- `F = 0.8645`, приріст `+0.0714`;
- latency `3399.3 ms`, практично така сама як baseline;
- FPS `6.36`, практично такий самий;
- VRAM `733.4 MB`, без вимірюваного приросту.

Найпростіший майже еквівалентний кандидат `box_centroid`:

- `J&F = 0.8847`;
- приріст `+0.0720`;
- latency `3424.7 ms`;
- він має на один prompt менше, ніж box + positive + negative, тому є кращим кандидатом за принципом простоти.

## Парний аналіз по sequence

Для `box_centroid - centroid` різниця `J&F` була позитивною на 7 із 10 sequence. Середній приріст — `+0.0720`, медіанний — `+0.0073`. Два великі прирости походять із `bike-packing` і `bmx-trees`, тому середнє частково залежить від складних sequence.

Детермінований bootstrap по 10 sequence дав приблизний 95% CI `[+0.0073, +0.1460]`. Це підтримує практично позитивний ефект у цьому test set, але не замінює повторення на повній DAVIS validation set і на незалежному dataset.

Для `box_positive_negative` різниця була позитивною на 8 із 10 sequence, середній приріст `+0.0719`, приблизний CI `[+0.0060, +0.1473]`.

## Аналітичні висновки

### Висновок 1: box prompt істотно кращий за centroid point

У цьому semi-supervised сценарії box дає приблизно +6.15 pp `J&F` проти однієї centroid point. Комбінація box + centroid дає +7.20 pp. Це логічно узгоджується з API SAM2: офіційний predictor підтримує points і boxes, а box задає просторову межу об'єкта.

### Висновок 2: додаткові points мають diminishing returns

Дві або три позитивні точки кращі за одну точку, але гірші за tight box + centroid. Три points погіршили результат порівняно з двома points. Це вказує, що важлива не кількість prompts, а правильний геометричний prior.

### Висновок 3: надмірне розширення box шкідливе

`box_expand_05` і `box_expand_10` гірші за tight box. Розширення 10% із negative point частково повертає якість, але не досягає `box_centroid`. Агенту варто оптимізувати не лише площу box, а й розташування corrective points.

### Висновок 4: якість можна покращити без суттєвої ціни inference

Усі prompt policies мають близький порядок latency і VRAM. Отже, виявлений приріст є quality improvement на рівні interaction policy, а не результатом збільшення моделі чи пам'яті.

### Висновок 5: це покращення системи, не checkpoint

Box policies використовують більше інформації від користувача, ніж одна точка. Тому результат не доводить, що SAM2.1 Hiera-Tiny став кращим як модель. Він доводить, що autoresearch може оптимізувати політику подачі prompt-ів до frozen model.

Для чесного порівняння “за однакової кількості інформації” наступною серією потрібно:

- порівняти лише policies з однаковим prompt budget;
- додати випадкове розташування точок;
- додати noisy boxes;
- виміряти якість після однакової кількості human interactions.

## Обмеження

1. Оцінено лише перші 20 кадрів кожної sequence.
2. Test set містить 10 із 30 validation sequence.
3. Prompts отримані з ground truth першого кадру; це стандартно для semi-supervised VOS, але не моделює помилки людини.
4. `J` є IoU-оцінкою, а `F` реалізовано локальним boundary evaluator із tolerance 2 pixels. Для фінального benchmark потрібно використати офіційний DAVIS evaluator.
5. Одна модель, один checkpoint і один детермінований run на policy.
6. Bootstrap CI має лише 10 незалежних sequence і тому є орієнтовним.
7. Optional SAM2 CUDA post-processing extension не скомпілювався; SAM2 продовжив inference, але цей факт потрібно врахувати при повторенні.

## Зв'язок із джерелами

- Офіційний SAM2 README описує `SAM2ImagePredictor`, `SAM2VideoPredictor`, points, boxes і propagation.
- SAM2 paper описує streaming memory та promptable segmentation у відео.
- DAVIS 2017 офіційно використовує region similarity `J` і contour accuracy `F`, а також `J&F` як агрегат.
- `karpathy/autoresearch` задає протокол гіпотеза → запуск → метрика → keep/discard.
- `modded-nanogpt` демонструє цінність фіксованих benchmark rules, журналювання і повторної перевірки.

Посилання на всі джерела зібрані в [sam2-sources.md](sam2-sources.md).

## Рішення

- `box_centroid`: **keep as primary candidate**;
- `box_positive_negative`: **keep as secondary candidate**, якість практично ідентична;
- `box`, `box_expand_05`, `box_expand_10`, `two_positive`, `three_positive`, `box_expand_10_negative`, `box_two_positive`: **discard for the current objective** або залишити як ablation records;
- наступна перевірка: повна DAVIS validation set, офіційний evaluator, noisy/user prompts і SA-V val transfer test.
