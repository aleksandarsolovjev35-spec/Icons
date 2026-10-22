# Как получить все иконки 256×256

## Windows — двойной клик

Нажми **`run_all.bat`**. Скрипт сам:

1. найдёт Python (нужен 3.10+; при установке отметь «Add Python to PATH»);
2. создаст окружение `.venv` и поставит `numpy` + `opencv-contrib-python-headless`;
3. возьмёт модель апскейла (FSRCNN уже лежит в репозитории — интернет нужен только
   для установки пакетов);
4. нарежет **все** листы, почистит и поднимет каждую иконку до 256×256;
5. соберёт превью в `samples/` и откроет папку с результатом.

Результат: **`icons_256/<папка>/<лист>__slotNN.png`** — 11 300 файлов, ~6 минут.

## macOS / Linux

```bash
chmod +x run_all.sh
./run_all.sh
```

## Полезные варианты

```bat
run_all.bat --limit 5          :: только первые 5 листов (проверка)
run_all.bat --slots 3          :: по 3 иконки с листа
run_all.bat --smooth           :: EDSR вместо FSRCNN (медленнее, чуть мягче)
run_all.bat --out my_folder    :: сложить результат в другую папку
```

Напрямую, без батника:

```bash
python3 tools/run_all.py
```

## Что происходит внутри

| шаг | инструмент |
|---|---|
| поиск рамок на листе | `tools/detect_frames.py` |
| вырез 1:1 с обводкой | `tools/cut_icons.py` |
| чистка от JPEG-шума | `tools/clean_icon.py` (`--preset normal`) |
| апскейл 256 | `tools/upscale_icon.py` (SR по яркости + доводка) |
| пакетный прогон | `tools/batch_all.py` |
| превью и отчёт | `tools/make_batch_previews.py` |

Превью после прогона: `samples/batch_contact_sheet.jpg`,
`samples/batch_montage.jpg`, отчёт — `samples/batch_summary.csv`.

## Окно и прогресс

Батник печатает весь прогресс (шаги 1/5…5/5, по строке на каждый лист) и в
конце **сам делает паузу** — «Press any key to close this window». Окно можно
свернуть, работа продолжится.

Весь вывод параллельно пишется в **`run_log.txt`** в корне: если окно всё же
закрылось, открой этот файл — там будет видно, на каком шаге остановилось.

## Если что-то пошло не так

- **«Python не найден»** — поставь Python 3.10+ и включи «Add Python to PATH».
- **Ошибка `dnn_superres`** — батник сам переустановит contrib-сборку opencv;
  вручную: `pip install --force-reinstall opencv-contrib-python-headless`.
- **Окно открылось и сразу закрылось** — раньше так и было: файл лежал в
  репозитории с Unix-переводами строк, а `cmd.exe` такой батник не дочитывает.
  Исправлено: `run_all.bat` хранится с CRLF, за это отвечает `.gitattributes`.
  Если у тебя старая копия — обнови репозиторий (`git pull`), либо почини файл
  в PowerShell:
  ```
  $p="$PWD\run_all.bat"; $c=Get-Content $p -Raw; [IO.File]::WriteAllText($p, ($c -replace "`r?`n","`r`n"))
  ```
- **В консоли «кракозябры» или падение `UnicodeEncodeError`** — было из-за
  рассинхрона кодировок: cmd.exe работает в cp866, а Python брал cp1251, и
  символ `->` (U+2192) в cp1251 отсутствует. Исправлено: `tools/_console.py`
  спрашивает у системы реальную кодовую страницу консоли и переключает вывод
  на неё, а непереводимые символы заменяет на `?` вместо падения.
- **`--smooth` не качает EDSR** — скачай `EDSR_x4.pb`
  (<https://github.com/Saafke/EDSR_Tensorflow/tree/master/models>)
  и положи в `tools/models/`.
