# Інструкція для наступного реального запуску

## Передумови

Потрібні:

- Python 3.10+;
- `python3-venv` або інший user-level environment manager;
- PyTorch >= 2.5.1;
- torchvision >= 0.20.1;
- CUDA, сумісна з PyTorch;
- SAM2 repository;
- DAVIS або інший dataset із ground-truth masks.

Поточний host має NVIDIA RTX 4070 Ti з 12 GB VRAM, тому починати слід із `sam2.1_hiera_tiny`, а не Large.

## Bootstrap

```bash
sudo apt install python3.12-venv
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install torch torchvision
.venv/bin/pip install -e /path/to/sam2
```

Checkpoint і config потрібно брати з офіційного SAM2 repository. Для першого run використовувати `sam2.1_hiera_tiny`.

## Протокол запуску

1. Зафіксувати baseline commit, checkpoint SHA-256 і dataset manifest.
2. Запустити baseline із незмінними prompts.
3. Запустити 10 кандидатів із таблиці в `sam2-iteration-log.md`.
4. Зберігати кожен run у власній git-гілці або commit.
5. Після кожного run рахувати метрики з `sam2-metric-protocol.md`.
6. Кожного переможця повторити тричі.
7. Лише після цього відкрити held-out test.
8. Перевірити переносимість на Hiera-Small.

## Мінімальний acceptance gate

Кандидат приймається, якщо на held-out test:

- `J&F` зростає щонайменше на 1 pp у quality-first режимі;
- або latency падає на 20% при падінні `J&F` не більше 0.5 pp;
- результат повторюється на 3 seeds;
- немає критичного зростання failure rate;
- LLM-оцінка не суперечить pixel-level metrics.

## Заборонені shortcuts

- Не оцінювати переможця лише за одним кадром.
- Не показувати held-out test агенту під час пошуку.
- Не змінювати dataset після перегляду результатів.
- Не називати LLM-візуальну оцінку ground truth.
- Не порівнювати latency на різних GPU без нормалізації.
