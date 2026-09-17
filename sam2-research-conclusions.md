# Фінальні висновки дослідження

Дата: 2026-08-28

## Відповідь на питання

Так, підхід `autoresearch` має сенс для SAM2, але його потрібно перенести з оптимізації language-model training на оптимізацію всієї системи promptable video/image segmentation.

Найкраща одиниця пошуку — комбінація:

- prompt policy;
- mask selection;
- memory/propagation policy;
- post-processing;
- inference configuration;
- fine-tuning, якщо є достатньо розмічених даних.

## Головний висновок

LLM повинен бути planner і code generator, але не єдиним evaluator. Ground truth evaluator має автоматично рахувати `J&F`, IoU, Dice, boundary F1, temporal consistency, failure rate, latency і VRAM.

Для SAM2 це сильніший дизайн, ніж просити LLM визначати, яка маска "виглядає краще". Візуальна оцінка LLM може бути корисною як допоміжний сигнал для класифікації помилок, але її потрібно калібрувати проти pixel-level metrics.

## Позитивні докази

У межах цієї сесії локальний функціональний запуск отримано: SAM2.1 Hiera-Tiny імпортується і виконує image inference на RTX 4070 Ti. На `truck.jpg` із positive point модель повернула 3 маски; найкращий internal score становив `0.9531`, latency — `256.92 ms`, peak VRAM — `501.8 MB`. Ground-truth маски для цього прикладу немає, тому це smoke baseline, а не доказ покращення segmentation quality.

Позитивна методологічна підтримка є зовнішньою:

- SAM2.1 у власному benchmark порівнюється краще за попередні SAM2 checkpoints;
- автор `autoresearch` повідомляв про приблизно 20 автономно знайдених покращень nanochat, які перенеслися на більшу модель;
- community `modded-nanogpt` демонструє, що довгі серії контрольованих benchmark-ітерацій можуть давати значний ефект.

Ці факти підтверджують правдоподібність підходу, але не є вимірюванням покращення SAM2 у цьому workspace.

## Перший локальний ground-truth baseline

DAVIS 2017 TrainVal 480p завантажено, ZIP integrity перевірено, dataset містить 90 sequence, 6208 кадрів і 6208 annotations. SAM2.1 Hiera-Tiny протестовано на перших 20 кадрах п'яти validation sequence: aggregate `J=0.7064`, `F=0.7229`, `J&F=0.7147`, середня latency `4232.8 ms`, `5.22 FPS`, peak VRAM `743.7 MB`.

Це baseline для майбутніх змін. Він не є ще доказом переваги autoresearch, бо порівняння з покращеним кандидатом не виконувалося.

## Перша контрольна ітерація

Кандидат із `offload_video_to_cpu=False` мав такий самий результат `J&F=0.7147`, latency `4213.0 ms` проти `4232.8 ms` у baseline і peak VRAM `1603.1 MB` проти `743.7 MB`. Покращення latency на 0.5% при збільшенні VRAM на 115.8% не проходить acceptance gate, тому кандидат відхилений. Це підтверджує, що оцінювати потрібно не тільки якість, а й Pareto trade-off ресурсів.

## Найперспективніша перша серія

1. Baseline SAM2.1 Hiera-Tiny на DAVIS subset.
2. Box expansion у межах 0%, 2%, 5%, 10%.
3. Додавання corrective positive/negative points за низькою confidence.
4. Вибір маски за stability score замість першого кандидата.
5. Connected-component і hole filtering.
6. Keyframe re-prompting при падінні temporal consistency.
7. Adaptive memory eviction.
8. BF16 проти FP16/FP32.
9. Encoder compilation і кешування image features.
10. Перенесення переможця на Hiera-Small і held-out test.

## Рішення про подальше дослідження

Дослідження варто продовжити після встановлення runtime. Першою ціллю має бути не fine-tuning ваг, а дешевий inference-loop: він дасть багато ітерацій, простіший rollback і менший ризик зіпсувати checkpoint.

Необхідно зберігати raw masks, commit hashes і конфігурацію кожного запуску. Переможець приймається лише після повторів на held-out test і перевірки trade-off між якістю та швидкістю.
