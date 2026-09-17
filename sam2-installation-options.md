# Варіанти встановлення SAM2 runtime

Дата: 2026-08-28

## Стан поточного середовища

- Ubuntu 24.04 у WSL2.
- NVIDIA GeForce RTX 4070 Ti, 12 GB VRAM.
- NVIDIA driver 610.88.
- Docker 29.1.3.
- Docker Compose 5.4.0.
- Docker daemon працює через Docker Desktop.
- У Docker доступний runtime `nvidia`.
- GPU passthrough перевірено: офіційний CUDA n-body container побачив RTX 4070 Ti і завершив benchmark успішно.
- Python 3.12 є, але `pip` і `python3.12-venv` відсутні.

## Порівняння

| Критерій | Docker-контейнер | Бібліотеки локально |
|---|---|---|
| Відтворюваність | Висока: фіксуються image, CUDA, PyTorch і SAM2 | Середня: залежності можуть конфліктувати |
| Початкова складність | Потрібен готовий Dockerfile/image і GPU passthrough | Потрібен один privileged bootstrap-крок |
| Вплив на host | Мінімальний, код і залежності ізольовані | Пакети живуть у `.venv`, але системний пакет потрібен |
| Швидкість ітерацій Python-коду | Добра через bind mount, але є container overhead | Найзручніша для debugger та IDE |
| CUDA compatibility | Контролюється image; driver лишається на host | Залежить від локальних wheels, driver і toolkit |
| Пам'ять/диск | Image може зайняти кілька GB | Wheels і checkpoints також займають кілька GB |
| Ризик зламати систему | Низький | Низький у venv, вищий при system-wide pip |
| SAM2 training/fine-tuning | Зручно для чистого повторного запуску | Зручно для швидкої розробки та налагодження |
| Рекомендація для цього проєкту | Основний benchmark runtime | Development runtime після bootstrap |

## Варіант A: Docker

### Чому це підходить

SAM2 потребує PyTorch >=2.5.1, torchvision >=0.20.1, CUDA-сумісного середовища і може компілювати CUDA extension. Контейнер дозволяє зафіксувати ці версії та повторити benchmark без залежності від локального Python.

Docker Desktop і NVIDIA runtime вже знайдені на host. GPU passthrough підтверджено тестовим CUDA-контейнером перед SAM2.

### Перевірка GPU

```bash
docker run --rm --gpus=all \
  nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

Це рекомендований Docker Desktop smoke test. Якщо образ недоступний або команда завершується помилкою, потрібно перевірити WSL2 integration у Docker Desktop і актуальність Windows NVIDIA driver.

### Орієнтовна структура

```text
sam2-runtime/
  Dockerfile
  compose.yaml
  requirements.txt
  benchmarks/
  checkpoints/
  data/
  results/
```

### Принципи image

- базуватися на NVIDIA CUDA runtime/devel image, сумісному з host driver;
- встановити PyTorch із CUDA wheels;
- встановити SAM2 із зафіксованого git commit;
- монтувати repository, `data/`, `checkpoints/` і `results/` через volumes;
- не вбудовувати checkpoint у image;
- запускати non-root user, якщо це не конфліктує з CUDA extension build.

### Переваги

- однакове середовище для всіх autoresearch runs;
- легко зберегти image digest;
- простий rollback до попередньої версії;
- ізоляція від системного Python;
- зручний запуск на іншому GPU host.

### Недоліки

- первинне завантаження image повільне;
- потрібно розібратися з bind mounts, user permissions і CUDA passthrough;
- IDE debugging трохи складніший;
- container не усуває залежність від NVIDIA driver на host.

## Варіант B: локальний venv

### Bootstrap

Поточний host не має `ensurepip`, тому потрібен privileged крок:

```bash
sudo apt update
sudo apt install -y python3.12-venv python3-pip
```

Після цього:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install torch torchvision
.venv/bin/pip install -e /path/to/sam2
```

Checkpoint SAM2.1 Hiera-Tiny і відповідний config потрібно завантажити з офіційного [SAM2 repository](https://github.com/facebookresearch/sam2).

### Переваги

- найпростіший workflow для локальної розробки;
- VS Code одразу бачить interpreter і debugger;
- швидше змінювати Python wrapper та evaluator;
- немає окремого Docker image lifecycle.

### Недоліки

- PyTorch/CUDA/SAM2 версії треба контролювати вручну;
- оновлення пакетів може змінити benchmark;
- залежності SAM2 можуть конфліктувати з іншими проєктами;
- без venv не можна рекомендувати system-wide pip.

## Рекомендоване рішення

Для цього дослідження рекомендую **гібридний порядок**:

1. GPU passthrough уже підтверджено CUDA smoke test.
2. Запустити baseline і всі benchmark-ітерації в контейнері.
3. Кодувати evaluator і prompt policies через bind-mounted workspace.
4. Зберігати image digest, checkpoint hash і results.
5. За потреби встановити локальний `.venv` для зручного налагодження, але не використовувати його як authoritative benchmark environment.

Це мінімізує ризик, що агент покращує результат через випадкову зміну версії бібліотеки або CUDA kernel, а не через власну ідею.

## Поточний статус

Docker GPU path готовий до підготовки SAM2 image. Локальний venv не створений через відсутність `python3.12-venv`; виконання `apt install` потребує дії користувача з sudo-привілеями. Для SAM2 залишаються наступні кроки: створити image, встановити SAM2, завантажити Hiera-Tiny checkpoint і запустити benchmark.

## Джерела

- [SAM2 official installation](https://github.com/facebookresearch/sam2#installation)
- [SAM2 official README](https://github.com/facebookresearch/sam2)
- [Docker Desktop GPU support for Windows/WSL2](https://docs.docker.com/desktop/features/gpu/)
- [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
