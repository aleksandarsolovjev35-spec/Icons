# Style guide

Все иконки библиотеки рисуются одним промптом-шаблоном. Отклонения от
шаблона ломают визуальную согласованность коллекции, поэтому шаблон
зафиксирован здесь и используется генератором дословно.

## Промпт-шаблон

```
Single game inventory icon, painterly digital art, three-quarter view from
slightly above: {SUBJECT}. Bold readable silhouette, simple readable shapes,
rich saturated colors, glossy material shading, soft rim light from the upper
left, subtle warm bounce light, subtle dark edge definition, high contrast,
centered, subject fills about 85 percent of the square frame, isolated on a
seamless pure white studio background, no shadow, no ground plane, no
reflection, no text, no border, no frame, no watermark, no UI, no extra
objects, no hands
```

`{SUBJECT}` — описание предмета: форма, материал, цвет, деталь, которая
делает силуэт узнаваемым (пробка и пузырьки у флакона, лента у свитка,
замо́к у сундука).

## Почему именно так

| Решение | Причина |
|---------|---------|
| `three-quarter view from slightly above` | единый ракурс; на 40 px силуэт читается лучше, чем в профиль |
| `bold readable silhouette` | в листе инвентаря иконка живёт 40–64 px, детали там не читаются |
| `simple readable shapes` | противовес к детализированному референсу из листов `RPG Loot Icons 01` |
| `subtle dark edge definition` | тёмный контур отделяет объект от листа, особенно на светлом фоне |
| `no shadow, no ground plane, no reflection` | тень и рефлекс не вырезаются чисто и дают серый ореол на прозрачном фоне |
| `pure white studio background` | детерминированное выделение альфы по формуле `C = aF + (1-a)W` |
| `subject fills about 85 percent` | кадрирование; фактический размер всё равно нормализуется скриптом |

## Постобработка

`tools/extract_alpha.py` приводит кадр к единому виду:

1. **Вырезание.** Альфа восстанавливается из тёмного канала, фон и контактная
   тень удаляются заливкой от границы, внутренние светлые области
   (бумага, блики на стекле) заливаются обратно в силуэт.
2. **Снятие ореола.** Полупрозрачный край разворачивается обратно из белого
   (`F = (C - (1-a)W) / a`), чтобы кромка не была белой.
3. **Нормализация.** Объект вырезается по bbox, масштабируется так, чтобы
   занимать **84 %** квадрата, и центрируется. Это то, что делает набор
   шаблонов: два флакона разного размера в исходнике станут одинаковыми по
   визуальному весу.
4. **Экспорт.** Мастер 512×512 PNG (`src/processed/`), в модуль — 200×200 WebP
   с альфа-каналом (Foundry показывает иконку до 64 px, но на 2×-экранах
   нужно больше).

## Правила приёмки

* Силуэт читается в 40 px (проверяется контакт-листом `preview/`).
* Нет белого ореола и серой кляксы на прозрачном фоне.
* Свет сверху-слева, та же тональность и насыщенность, что у соседей.
* Объект занимает одинаковую долю кадра.
