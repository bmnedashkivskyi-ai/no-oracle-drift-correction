# Протокол метричної оцінки autoresearch + SAM2

## Мета

Порівнювати baseline SAM2 із кожною зміною агента на однакових даних, prompts, checkpoint, hardware і random seeds.

## Рівні оцінки

### Pixel-level

Для прогнозованої маски `P` і ground-truth маски `G`:

- IoU/Jaccard: `|P ∩ G| / |P ∪ G|`;
- Dice/F1: `2|P ∩ G| / (|P| + |G|)`;
- Boundary F-score із tolerance у 1, 2 і 5 пікселів;
- area error: відносна похибка площі маски.

Для кожного об'єкта рахувати середнє і медіану, а не лише micro-average, щоб великі об'єкти не приховували провали на малих.

### Відео

Для відеопослідовностей використовувати:

- `J`: region similarity, середній IoU;
- `F`: contour accuracy, boundary F-score;
- `J&F = (J + F) / 2`;
- temporal consistency: IoU між маскою поточного кадру і warped-маскою попереднього кадру;
- identity/object preservation для multi-object tracking;
- failure rate: частка послідовностей із втратою об'єкта.

### Система

Фіксувати:

- end-to-end latency і окремо encoder/prompt/propagation latency;
- frames per second;
- peak VRAM і RAM;
- model size;
- кількість prompt interactions до заданої якості;
- energy/cost, якщо доступні дані GPU.

## Основна цільова функція

Для quality-first режиму:

`score = J&F - 0.05 * max(0, latency_ratio - 1) - 0.02 * max(0, vram_ratio - 1)`

Коефіцієнти треба затвердити до запуску і не змінювати після перегляду результатів. Для production-сценарію краще використовувати Pareto frontier замість довільного зведення до одного числа.

Для speed-first режиму кандидат приймається, якщо:

- `J&F` не падає більш ніж на 0.5 процентного пункту;
- latency скорочується щонайменше на 20%;
- peak VRAM не зростає більш ніж на 10%.

## Експериментальний split

- development: 60% послідовностей;
- validation: 20%, використовується агентом;
- held-out test: 20%, не показується агенту до завершення серії.

Validation потрібно розділяти за відео, а не випадково за кадрами. Інакше сусідні кадри створять leakage.

## Повторюваність

Кожен значущий кандидат перевіряти мінімум на 3 seeds або 3 prompt placements. Зберігати:

- commit hash;
- SAM2 checkpoint і config;
- версії PyTorch/CUDA;
- dataset manifest і SHA-256;
- prompts;
- raw predictions;
- JSON/TSV з метриками;
- stdout, stderr і peak resource measurements.

## LLM-взаємодія

LLM може:

1. сформулювати наступну гіпотезу;
2. змінити prompt policy або Python wrapper;
3. прочитати агреговані результати;
4. запропонувати збереження/відкат зміни;
5. пояснити типові failure cases.

LLM не може бути єдиним оцінювачем. Візуальна оцінка LLM зберігається як secondary signal і перевіряється кореляцією зі справжнім `J&F`, IoU та boundary F-score.

## Критерій позитивного результату

Публікувати покращення лише якщо воно:

- перевищує baseline на held-out test;
- має bootstrap 95% confidence interval, що не перетинає нуль або має практично значущий ефект;
- повторюється на трьох seeds;
- не погіршує інші ключові категорії;
- має зрозуміле resource trade-off.

## Мінімальний запис одного run

```text
run_id, parent_commit, candidate_commit, checkpoint, dataset_version,
prompt_policy, seed, J, F, J&F, mean_iou, boundary_f1,
latency_ms, fps, peak_vram_mb, interactions, status, notes
```
