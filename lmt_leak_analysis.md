# Analiza leaku informacji w pierwszej wersji LMT

## Setup

W oryginalnym LMT (`equivariant_modules_lmt.py`):

1. MA encoder dostaje **czyste (niezamaskowane)** piksele wejściowe.
2. MA produkuje feature mapę 57x57 z 16 trivial scalarami.
3. `torch.where(mask, mask_token, ma_features)` nadpisuje **tylko nominalne 4x4 feature patche**.
4. PM encoder i downstream dostają feature mapę z tokenami w nominalnych pozycjach.

Problem: feature piksele **poza** nominalną maską, ale **w zasięgu receptive field MA od zamaskowanego regionu**, zawierają informację o oryginalnych pikselach, które miały być zamaskowane.

## Receptive field MA encodera

Architektura MA (`ma_layers_blocks: [1]`, `include_stem: true`):

| Warstwa | Kernel | Stride | Padding | Promień (feat px) |
|---------|--------|--------|---------|-------------------|
| Stem (`R2Conv`) | 3x3 | 2 | 1 | 1 |
| BLConvNeXt depthwise (`R2Conv`) | 7x7 | 1 | 3 | 3 |
| Regular2Trivial | 1x1 | 1 | 0 | 0 |
| pw_up / pw_down (1x1) | 1x1 | 1 | 0 | 0 |

**Efektywny promień RF = 1 + 3 = 4 feature piksele.**

## Ile feature pikseli jest "leaky"

Dla zamaskowanego input patcha 8x8 (= 4x4 w feature space):

- **Nominalny patch** (nadpisany tokenem): 4x4 = **16** feature pikseli -- brak leaku.
- **Ring leaku** (NIE nadpisany tokenem, ale zawiera info o masce): pierścień o grubości 4 wokół patcha = (4+2·4)x(4+2·4) - 4x4 = 12x12 - 16 = **128** feature pikseli.

**Stosunek leak do mask: 128 / 16 = 8x więcej feature pikseli z wyciekiem niż z tokenem.**

## Siła leaku w zależności od odległości

Nie wszystkie 128 pikseli leakują jednakowo. Im dalej od patcha, tym słabiej:

### Odległość 1 (ring od stemu)

Feature piksel w odległości 1 od brzegu nominalnego patcha. Jego kernel stemu (3x3, stride 2) bezpośrednio czyta **1-2 zamaskowane piksele wejściowe** z 3 pikseli w kernelu.

- Kontaminacja: **~33-66%** wag kernela operuje na zamaskowanych pikselach.
- Siła leaku: **wysoka** -- wartość feature piksela jest bezpośrednio zależna od zamaskowanych danych.
- Liczba pikseli w tym ringu: (4+2)x(4+2) - 4x4 = 36 - 16 = **20** feature pikseli.

### Odległość 2-4 (ring od depthwise 7x7)

Feature piksele w odległości 2-4 od brzegu. Ich wartość zależy od zamaskowanych danych **pośrednio**: depthwise 7x7 miesza 49 stem-output pikseli, z których tylko część jest skontaminowana przez stem.

- Odległość 2: depthwise kernel sięga do 1 skontaminowanego stem-piksela z 49 → ~2% wag.
- Odległość 3: depthwise kernel sięga dalszych, mocniej rozcieńczonych pozycji → <1%.
- Odległość 4: minimalny leak, na granicy szumu numerycznego.
- Liczba pikseli: (12x12) - (6x6) = 144 - 36 = **108** feature pikseli.

### Podsumowanie per-ring

| Ring | Odległość (feat px) | Źródło | Pikseli | Siła leaku |
|------|---------------------|--------|---------|------------|
| Nominal | 0 (wewnątrz patcha) | -- | 16 | 0% (token) |
| Ring 1 | 1 | stem 3x3 | 20 | ~33-66% |
| Ring 2 | 2 | depthwise 7x7 | 36 | ~5-10% |
| Ring 3 | 3 | depthwise 7x7 | 36 | ~1-2% |
| Ring 4 | 4 | depthwise 7x7 | 36 | <1% |

## Efektywny leak na poziomie całego obrazu

Parametry: `spatial_masking_ratio: 0.6`, `mask_patch_size: 8`, feature map 57x57.

- Grid patchy: 15x15 = 225 patchy.
- Zamaskowanych: ~135 patchy (60%).
- Nominalna maska (token): 135 x 16 = **2160** feature pikseli.
- Ring 1 (silny leak): do 135 x 20 = 2700 pikseli, ale ringi sąsiednich patchy się nakładają. Efektywnie: ~**800-1200** unikalnych feature pikseli z silnym leakiem (~33-66%).

Przy gęstym maskowaniu (60%) sąsiednie patche mają zazwyczaj nakładające się ringi, więc **prawie każdy feature piksel w okolicy zamaskowanego regionu jest albo tokenem, albo leaky**. "Czyste" feature piksele (zerowy leak) to głównie te w środku dużych niezamaskowanych regionów.

## Co to oznacza dla modelu

PM encoder (12 bloków, rosnące RF) widzi feature mapę, w której:
- ~60% to mask token (nominalna maska).
- ~10-15% to feature piksele z **silnym** leakiem (ring 1) -- PM może z nich odczytać fragmenty zamaskowanych danych.
- Reszta: słaby leak lub czyste.

PM ma wystarczający receptive field, żeby **sięgnąć z wewnątrz maski do ring-1 pikseli** i odzyskać część zamaskowanej informacji bez "prawdziwego" inferowania z kontekstu. To potencjalnie osłabia sygnał uczący się rekonstrukcji z kontekstu -- model "oszukuje" korzystając z boundary leaku zamiast uczyć się sensownych reprezentacji.

## Porównanie: LMT vs LMT no-leak

| | LMT (oryginał) | LMT no-leak |
|---|---|---|
| Input do MA | czysty | zerowane patche |
| Ring 1 leak | ~33-66% | 0% (MA widzi zera) |
| Ring 2-4 leak | ~1-10% | 0% |
| Token pokrywa | 4x4 nominalne | 4x4 nominalne |
| Boundary artifact | brak | tak (MA przetworzyło zera) |

W wersji no-leak: MA przetwarza zera w zamaskowanych regionach. Feature piksele w ringu 1 nadal "widzą" coś w swoim kernelu, ale to co widzą to **zera, a nie oryginalne dane**. Zero bitów informacji o oryginale. Jedyny sygnał to "tu był brzeg maski" (boundary artifact), co nie jest leakiem -- to sygnał pozycyjny analogiczny do positional embeddings w MAE.
