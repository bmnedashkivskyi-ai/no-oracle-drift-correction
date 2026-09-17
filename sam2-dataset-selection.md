# Вибір відкритого dataset для SAM2 autoresearch

Дата: 2026-08-28

## Обраний основний dataset

**DAVIS 2017 TrainVal, semi-supervised, 480p**

Офіційне завантаження:

https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip

Офіційна сторінка dataset/evaluation:

https://davischallenge.org/davis2017/code.html

Офіційний Python evaluator:

https://github.com/davisvideochallenge/davis2017-evaluation

## Чому саме DAVIS 2017

1. Має відеокадри та pixel-level annotations.
2. Містить об'єкти, що рухаються, деформуються, частково перекриваються і зникають.
3. Є стандартні метрики `J`, `F` та `J&F`.
4. Доступний прямий архів без ручного погодження доступу.
5. Версія 480p економніша за full-resolution і придатна для RTX 4070 Ti з 12 GB VRAM.
6. Невеликий за сучасними мірками, тому зручний для десятків швидких autoresearch iterations.
7. Semi-supervised сценарій відповідає SAM2: подати prompt на першому кадрі та оцінювати propagation.

Офіційна сторінка DAVIS зазначає, що метрики рахуються на 480p, навіть якщо для досліджень можна використовувати full resolution.

## Розподіл для дослідження

DAVIS TrainVal потрібно розділити за video sequence:

- `dev`: 60% послідовностей — розробка prompt policy;
- `validation`: 20% — рішення keep/discard для агента;
- `held-out test`: 20% — відкривається лише після завершення серії.

Не можна випадково ділити окремі кадри між split-ами, бо сусідні кадри створять leakage.

## Порівняння кандидатів

| Dataset | Переваги | Недоліки | Рішення |
|---|---|---|---|
| DAVIS 2017 TrainVal 480p | Компактний, відкритий, стандартні J/F, повні маски | Менш різноманітний за SA-V | Основний перший benchmark |
| SA-V val/test | 155 val і 150 test відео, ручні маски, створений для SAM2, evaluator у репозиторії SAM2 | Потрібне окреме завантаження, більший pipeline, access/data management | Другий етап для transfer test |
| SA-V train | 50,583 відео і 643K masklets, CC BY 4.0 | Надто великий для першого швидкого циклу; багато даних для fine-tuning | Пізній fine-tuning stage |
| YouTube-VOS | Великий і різноманітний benchmark, VOS/ VIS/RVOS tasks | Потрібно уважно перевірити доступ, формат і умови використання | Додатковий зовнішній test |
| COCO/SA-1B image subsets | Зручні image masks | Не оцінюють temporal propagation | Лише image-side ablation |

## Другий рекомендований dataset

Після DAVIS використати **SA-V val** як transfer test. Офіційний SAM2 repository вказує:

- 155 validation videos;
- 293 manually annotated masklets;
- frames у JPEG 24 fps;
- annotations у PNG 6 fps;
- evaluator для `J` і `F`;
- ліцензію відео й annotations CC BY 4.0.

SA-V val є найкращим тестом на те, чи покращення не є overfitting до DAVIS. Не слід використовувати його для вибору гіпотез, якщо він призначений як held-out transfer set.

## Завантаження

Поточний workspace успішно завантажив і розпакував офіційний архів DAVIS 2017 TrainVal 480p. Архів перевірено командою `unzip -tq` без помилок. SHA-256 архіву:

```text
e3d0b5b77c3d031b000a19e0e25e3e2cac65d183755601bc2cf066df1a2aa492
```

Розмір розпакованого dataset — приблизно 1.6 GB. Workspace містить його за шляхом:

```text
data/davis/DAVIS-2017-trainval-480p.zip
```

Розпаковані дані:

```text
data/davis/DAVIS/
```

Перевірена структура містить 90 sequence, 6208 JPEG frames і 6208 PNG annotations; split має 60 train та 30 validation sequence.

Після завершення потрібно перевірити:

```text
data/davis/DAVIS/
  JPEGImages/480p/<sequence>/<frame>.jpg
  Annotations/480p/<sequence>/<frame>.png
  ImageSets/2017/train.txt
  ImageSets/2017/val.txt
```

## Рекомендований evaluation protocol

Для кожної DAVIS validation sequence:

1. Взяти ground-truth prompt на першому кадрі.
2. Запустити SAM2 video predictor.
3. Propagate mask через усю послідовність.
4. Зберегти маски у DAVIS-compatible structure.
5. Порахувати `J`, `F`, `J&F`, decay і failure rate.
6. Повторити для кожної prompt policy.
7. Після вибору кандидата перевірити його на held-out test і SA-V val.

## Ліцензії та цитування

Перед публічним розповсюдженням результатів перевірити актуальні умови DAVIS. У науковій роботі слід цитувати DAVIS 2017 paper і DAVIS evaluation package. SA-V має окрему CC BY 4.0 ліцензію та власні citation requirements.

## Висновок

DAVIS 2017 TrainVal 480p — найкращий стартовий компроміс для цього workspace: він достатньо складний для перевірки video segmentation, має офіційну ground truth і evaluation, але не створює надмірного storage/compute бар'єра. SA-V val слід зберегти як незалежний transfer test, а SA-V train — лише для наступного етапу fine-tuning.
