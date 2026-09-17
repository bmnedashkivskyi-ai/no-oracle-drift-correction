# Джерела і матеріали

Дата перевірки: 2026-08-28

## Першоджерела SAM2

1. **SAM 2: Segment Anything in Images and Videos** — Nikhila Ravi et al., arXiv:2408.00714, 2024.
   https://arxiv.org/abs/2408.00714
   
   Основне джерело про promptable segmentation, streaming memory, відеоінференс і model-in-the-loop data engine. Автори повідомляють, що SAM2 використовує приблизно втричі менше взаємодій у відеосегментації, а для зображень є точнішим і швидшим за SAM.

2. **Офіційний код SAM2**
   https://github.com/facebookresearch/sam2
   
   Містить SAM2.1 checkpoints, image/video predictor APIs, training code, notebooks і benchmark guidance.

3. **SAM2.1 checkpoints**
   https://github.com/facebookresearch/sam2#model-description
   
   Доступні Hiera Tiny, Small, Base Plus і Large. У README вказані параметри приблизно від 38.9M до 224.4M і таблиця офіційних benchmark values.

4. **SAM2 training documentation**
   https://github.com/facebookresearch/sam2/tree/main/training
   
   Інструкції для training/fine-tuning на власних image/video datasets.

5. **SA-V dataset**
   https://ai.meta.com/datasets/segment-anything-video
   
   Великий датасет відеосегментації, створений за допомогою model-in-the-loop data engine. Ліцензійні та access restrictions потрібно перевірити перед використанням.

6. **DAVIS dataset and benchmark**
   https://davischallenge.org/
   
   Стандартний benchmark для video object segmentation; використовує region/contour metrics, зокрема J і F.

6a. **DAVIS 2017 evaluation package**
   https://github.com/davisvideochallenge/davis2017-evaluation
     
   Офіційний Python package для semi-supervised та unsupervised DAVIS 2017. Для фінального звіту слід прогнати саме цей evaluator, а не лише локальну lightweight реалізацію.

## Порівняльні та методологічні джерела

7. **Karpathy autoresearch**
   https://github.com/karpathy/autoresearch
   
   Патерн single-file agent loop: гіпотеза, commit, фіксований запуск, metric extraction, keep/discard і журнал.

8. **Karpathy nanochat**
   https://github.com/karpathy/nanochat
   
   Повніший LLM training/evaluation pipeline, з якого autoresearch бере багато компонентів ідей.

9. **Modded-NanoGPT**
   https://github.com/KellerJordan/modded-nanogpt
   
   Приклад community speedrun з фіксованим benchmark, статистичною перевіркою та історією великої кількості покращень. Корисний для дизайну правил, але його language-model metrics не можна напряму переносити на SAM2.

10. **SAM2 README: inference and model description**
    https://github.com/facebookresearch/sam2/blob/main/README.md
    
    Фіксує вимоги `Python >= 3.10`, `torch >= 2.5.1`, predictor APIs, `torch.compile`, checkpoint links і model license.

## Джерела зовнішніх позитивних контролів

11. **Karpathy: autoresearch results on nanochat**
    https://x.com/karpathy/status/2031135152349524125
    
    Автор описує приблизно 20 змін, знайдених автономним агентом, їх переносимість на більшу модель і покращення time-to-GPT-2. Це evidence for methodology, не доказ покращення SAM2.

12. **Karpathy: nanochat training improvement and automated iteration**
    https://x.com/karpathy/status/2029701092347630069
    
    Описано автоматичні ітерації над nanochat та роль dataset/optimization changes.

13. **Meta SA-V download terms and access form**
   https://ai.meta.com/datasets/segment-anything-video-downloads/
    
   Сторінка вимагає дані організації/дослідника, intended use і явне прийняття CC BY 4.0 перед download. У цьому дослідженні форма не заповнювалася вигаданими даними.

## Матеріали, які потрібно додати після реального запуску

- exact git commit SAM2;
- checkpoint SHA-256;
- dataset manifests і ліцензії;
- prompt annotations;
- raw masks і visualization montage;
- run logs;
- confidence intervals;
- hardware/software manifest;
- error taxonomy та ablation tables.

## Ліцензійна примітка

Офіційний SAM2 code, checkpoints і training code у README позначені Apache 2.0; окремі third-party компоненти можуть мати власні ліцензії. Ліцензії датасетів і зображень потрібно перевіряти окремо.
