# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**MagicBrain** - библиотека спайковых нейронных сетей (SNN) с ДНК-подобным кодированием и структурной пластичностью. Часть экосистемы StudyNinja для исследования когнитивных процессов обучения и адаптивных алгоритмов.

## Installation

```bash
cd /root/StudyNinja-Eco/projects/MagicBrain
pip install -e .
```

## Common Commands

### Running CLI

```bash
# Train a new model
magicbrain train --genome "30121033102301230112332100123" --steps 10000 --out model.npz

# Train from custom text file
magicbrain train --text /path/to/text.txt --steps 20000 --out model.npz

# Resume training from existing model
magicbrain train --load model.npz --steps 5000 --out model_v2.npz

# Generate text from trained model
magicbrain sample --model model.npz --seed "To be" --n 500 --temp 0.75

# Run self-repair benchmark
magicbrain repair --genome "30121033102301230112332100123" --damage 0.2
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_smoke.py

# Run with verbose output
pytest -v

# Run single test function
pytest tests/test_smoke.py::test_brain_init
```

### Development

```bash
# Install in editable mode with development access
pip install -e .

# Run from Python
python -c "from magicbrain import TextBrain; brain = TextBrain('30121033102301230112332100123', 50); print(brain.N)"
```

## Architecture

### Core Design: DNA-Encoded Spiking Neural Networks

MagicBrain реализует биологически инспирированную архитектуру с несколькими ключевыми компонентами:

#### 1. Genome System (`genome.py`)

**Геном как строка**: Нейросеть полностью определяется строкой символов '0'-'3' (base-4 кодирование).

**Декодирование**: Функция `decode_genome()` преобразует геном в 17 гиперпараметров:
- Размер сети (`N`): 256-832 нейронов
- Степень связности (`K`): 8-20 ближайших соседей
- Learning rate, decay rates, pruning parameters
- Модуляторные параметры (dopamine_gain, dopamine_bias)

**Важно**: Один геном = одна воспроизводимая архитектура. Изменение одного символа может существенно изменить поведение сети.

#### 2. Brain Architecture (`brain.py`)

**TextBrain** - основной класс спайковой сети для задач предсказания следующего символа.

**Ключевые компоненты**:

- **Activation state (`a`)**: Бинарная спайковая активность нейронов (N,)
- **Dual-timescale traces**:
  - `trace_fast`: Быстрая синаптическая память (decay ~0.92-0.94)
  - `trace_slow`: Медленная память для контекста (decay ~0.985-0.988)
- **Dual-weight system**:
  - `w_slow`: Долговременная синаптическая память (консолидация)
  - `w_fast`: Быстрая пластичность для обучения
- **Delayed buffers**: 5 временных буферов для моделирования задержек передачи сигнала (1-5 временных шагов)
- **Homeostatic threshold (`theta`)**: Адаптивный порог для поддержания целевой активности
- **Output readout**: Линейный слой `R` + bias `b` для отображения состояния сети в логиты

**Sparse activation**: На каждом шаге активно только `k_active` нейронов (~5% от N).

**E/I balance**: Часть нейронов помечена как тормозные (`is_inhib`), их веса всегда неположительные.

#### 3. Graph Structure (`graph.py`)

**Spatial graph**: Нейроны расположены в 3D пространстве (случайные координаты).

**Connectivity**:
- Каждый нейрон соединён с `K` ближайшими соседями (локальная связность)
- Дополнительно `p_long` * E случайных дальних связей (long-range connections)

**Axonal delays**: Задержки пропорциональны евклидовому расстоянию между нейронами (1-5 шагов).

**Importance**: Пространственная структура критична для формирования иерархических представлений.

#### 4. Learning Dynamics (`brain.py` методы)

**Forward pass** (`forward(token_id)`):
1. Сдвиг delay buffers
2. Вычисление входного сигнала: `x = delayed_now - theta + noise`
3. Sparse selection: выбор `k_active` нейронов с наибольшим входом
4. Обновление traces (fast и slow)
5. Рекуррентные связи: взвешенная активность пропагируется через граф с задержками
6. Формирование состояния: `state = a + alpha*trace_fast + beta*trace_slow`
7. Readout: `logits = state @ R + b`

**Learning** (`learn(target_id, probs)`):
1. Вычисление loss и gradient для выходного слоя
2. **Neuromodulation**: Расчёт dopamine на основе advantage (loss_ema - loss)
3. **Output learning**: Градиентный шаг для `R` и `b` (клиппинг для стабильности)
4. **Recurrent learning**: Hebbian-like правило с модуляцией:
   - `dW = lr * dopamine * advantage * pre * post`
   - `pre = trace_fast[src]`, `post = a[dst]`
5. **Consolidation**: Перенос `w_fast` в `w_slow` (медленная интеграция)
6. **Homeostasis**: Адаптация `theta` для поддержания target firing rate
7. **Pruning/Rewiring**: Периодическое удаление слабых связей и случайная реорганизация

**Критические аспекты**:
- Dopamine-modulated learning (reward-based plasticity)
- Dual weight system позволяет быстрое обучение с сохранением долговременной памяти
- E/I invariant: веса тормозных нейронов принудительно неположительные после каждого обновления

#### 5. Structural Plasticity

**Pruning**: Периодическое удаление `prune_frac` слабейших связей (по `|w_slow + w_fast|`)

**Rewiring**: Часть удалённых связей (`rewire_frac`) переподключается к случайным нейронам с новыми весами

**Self-repair capability**: Сеть может восстанавливаться после повреждения связей (`damage_edges()`), что тестируется в benchmark.

#### 6. Sampling and Generation (`sampling.py`)

**Temperature scaling**: Контроль разнообразия выходных токенов
**Top-k filtering**: Ограничение выбора k наиболее вероятными токенами
**Top-p (nucleus) sampling**: Динамический cutoff по кумулятивной вероятности

**Генерация**: Итеративный процесс forward + sample из распределения.

### Key Design Patterns

**State composition**: Финальное состояние сети - это weighted sum трёх компонент:
- Текущая активность (`a`)
- Быстрая память (`trace_fast`) с коэффициентом `alpha`
- Медленная память (`trace_slow`) с коэффициентом `beta`

**Sparsification**: Перед композицией traces обрезаются до top-m значений (`sparsify_topm`) для эффективности и биологической правдоподобности.

**Normalization**: Итоговое состояние нормализуется так, чтобы сумма элементов равнялась `state_sum_target` (обычно `k_active`).

## Code Organization

```
magicbrain/
├── brain.py         - Основной класс TextBrain со всей логикой SNN
├── genome.py        - Декодирование генома в параметры
├── graph.py         - Построение пространственного графа связей
├── sampling.py      - Генерация текста с температурой и фильтрацией
├── io.py            - Сохранение/загрузка моделей (npz формат)
├── utils.py         - Утилиты (softmax, sigmoid, sparsify)
├── cli.py           - CLI интерфейс (train, sample, repair)
└── tasks/
    ├── text_task.py - Построение vocab и training loop
    └── self_repair.py - Бенчмарк самовосстановления после повреждения
```

## Integration with StudyNinja

MagicBrain потенциально может использоваться для:

1. **Моделирование обучения студентов**: Спайковые сети с пластичностью как модель формирования знаний
2. **Адаптивные алгоритмы**: Исследование механизмов адаптации и self-repair для образовательных систем
3. **Нейронное моделирование**: Изучение когнитивных процессов через биологически правдоподобные сети

**Важно**: MagicBrain - это исследовательская библиотека, не production-сервис. Используется для экспериментов с нейроморфными подходами.

## File I/O Format

**Model files** (`.npz`):
- `genome_str`: Исходная строка генома
- `vocab`: JSON с `stoi` (str->int) и `itos` (int->str)
- `w_slow`, `w_fast`: Массивы весов синапсов
- `R`, `b`: Параметры выходного слоя
- `theta`: Гомеостатические пороги
- `meta`: Метаданные (step, timestamp, N, K)

## Testing Strategy

**Smoke tests** (`test_smoke.py`):
- Genome decoding correctness
- Brain initialization (shapes, dimensions)
- Forward pass (probability distribution validity)
- Learning step (loss computation)
- Save/load roundtrip
- E/I sign invariant (inhibitory weights stay non-positive)

**Для новых фич**: Добавлять тесты в `tests/` следующие паттернам smoke tests.

## Common Development Patterns

### Creating a new task

1. Создать файл в `magicbrain/tasks/your_task.py`
2. Определить функции для data preparation и training loop
3. Использовать паттерн из `text_task.py`:
   - Build vocab/encoding
   - Training loop с периодическим логгированием
   - Использовать `brain.forward()` и `brain.learn()`

### Adding CLI command

1. В `cli.py` добавить subparser
2. Определить аргументы
3. Добавить обработку в `main()` функцию
4. Импортировать необходимые модули из `magicbrain.tasks`

### Модификация архитектуры

**Критические точки**:
- `brain.py` методы `forward()` и `learn()` - ядро системы
- При изменении структуры убедиться в совместимости с `io.py` (save/load)
- После изменений параметров обновить `genome.py` decode logic
- Обновить тесты в `tests/test_smoke.py`

### Debugging Tips

**Print diagnostics** во время обучения:
- `brain.dopamine`: должен колебаться вокруг 0.5
- `brain.avg_theta()`: медленно растёт к positive values
- `brain.firing_rate()`: должен быть близок к `target_rate`
- `brain.mean_abs_w()`: средняя абсолютная величина весов

**Если loss не падает**:
- Проверить learning rate в геноме
- Убедиться что `k_active` не слишком маленький
- Проверить что dopamine модуляция работает (не зафиксирована на 0 или 1)

**Если генерация бессмысленна**:
- Убедиться что модель достаточно обучена (>10k steps на простом тексте)
- Проверить temperature параметр (0.7-0.9 обычно работает)
- Проверить что vocab правильно загружен

## Performance Considerations

**Memory**: Основная память - это граф связей (N*K edges) и trace arrays (N размера).

**Typical sizes**:
- N=384, K=12, vocab=50 → ~200KB model file
- Training: ~10k steps/min на CPU для небольших сетей

**Scaling**: Для больших N (>1000) операции с графом могут замедлиться. Рассмотрите sparse matrix operations.

## Key Parameters (via Genome)

**Наиболее важные для результата**:
- `N` и `K`: Определяют capacity сети
- `lr`: Скорость обучения (слишком большой → нестабильность)
- `k_active`: Количество активных нейронов (влияет на sparse representation)
- `alpha` / `beta`: Баланс между быстрой и медленной памятью
- `dopamine_gain` / `dopamine_bias`: Чувствительность к reward signal

**Для экспериментов начать с дефолтного генома**: `"30121033102301230112332100123"`
