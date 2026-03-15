# Звіт про генерацію B-матриці

Цей звіт описує створені матриці `B`, їхні технічні відмінності, як основні набори даних використовуються в проєкті, та де знаходяться кінцеві файли у форматах Parquet, CSV та Excel.

Для ознайомлення з формулами для кожної ознаки, кількістю порожніх значень (null) та детальними інтерпретаціями 52 стовпців у `B1_raw_train`, дивіться [b1_raw_train_null_report_uk.md](/D:/GitHub/transition-matrix-ecg/reports/b1_raw_train_null_report_uk.md).

## Сімейства матриць

- `B_raw` — це матриця, зрозуміла для лікаря, у її природних одиницях вимірювання. Вона зберігає порожні значення та залишає стовпці метаданих для звітності: `record_id`, `qtc_formula_code` та `split`.
- `B_fit` — це типізоване перетворення `B_raw`, яке використовується для підгонки переходу. Вона генерується за допомогою статистики лише з навчального набору даних і використовує специфічні для сімейства перетворення, реалізовані у [typed_transforms.py](/D:/GitHub/transition-matrix-ecg/src/tm_ecg/transition/typed_transforms.py).
- У поточному запуску кожна матриця `B_raw_*` має загалом `52` стовпці, тоді як кожна матриця `B_fit_*` має загалом `44` стовпці, оскільки метадані та цільові змінні, які не підлягають підгонці, виключаються або об'єднуються для моделювання.

## Дані

Як завантажити дані?

| Назва набору даних | Версія | Репозиторій PhysioNet | Репозиторій Kaggle |
| :--- | :---: | :--- | :--- |
| **PTB-XL** | `v1.0.3` | [Переглянути на PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) | [Переглянути на Kaggle](https://www.kaggle.com/datasets/garethwmch/ptb-xl-1-0-3) |
| **PTB-XL+** | `v1.0.1` | [Переглянути на PhysioNet](https://physionet.org/content/ptb-xl-plus/1.0.1/#files-panel) | [Переглянути на Kaggle](https://www.kaggle.com/datasets/antonymgitau/ptb-xl-a-comprehensive-ecg-feature-dataset) |
| **LUDB** | `v1.0.1` | [Переглянути на PhysioNet](https://physionet.org/content/ludb/1.0.1/) | Немає |

## Ролі наборів даних у цьому проєкті

| Набір даних | Роль у проєкті | Чому він використовується | Результуюче сімейство матриць | Поточна експортована реалізація |
| --- | --- | --- | --- | --- |
| `PTB-XL v1.0.3 + PTB-XL+ v1.0.1` | Масштабний еталонний набір даних для основного переходу від латентного простору до клінічних ознак. | Він забезпечує широкий масштаб когорти, офіційну структуру фолдів, 10-секундні сигнали у 12 відведеннях, а також додаткову інформацію PTB-XL+ для гармонізації ознак та верифікації вимірювань. | `B1_*` | `train=10000`, `val=2183`, `test=2198` |
| `LUDB v1.0.1` | Золотий стандарт набору даних з розміткою для виділення ознак, що зберігають морфологію, та для вторинної валидації пояснюваності. | Він містить ручні межі та піки для P/QRS/T, що робить його пріоритетним джерелом для створення високонадійних клінічних ознак, незважаючи на менший розмір. | `B2_*` | Поточна реалізація фолдів для виконання: `train=120`, `val=40`, `test=40` |

Отже, PTB-XL/PTB-XL+ та LUDB не є взаємозамінними у проєкті:

- `B1` — це масштабована еталонна матриця, яка використовується для відтворення методології матриці переходів на `10000` рядках на сучасній великій когорті ЕКГ.
- `B2` — це менша матриця з вищою точністю, яка використовується, коли якість ручної розмітки важливіша за кількість зразків.
- Шлях класифікатора та виділення латентних ознак ґрунтується на PTB-XL; LUDB використовується для створення другого, морфологічно обґрунтованого простору `B` для валідації пояснювального переходу.
- MIT-BIH залишається лише допоміжним і не є частиною сімейства базових артефактів `B1` або `B2`.

## Медичні протоколи та стандарти

Згідно з [AGENTS.md](/D:/GitHub/transition-matrix-ecg/AGENTS.md), матриці `B1_*` і `B2_*` мають спиратися на визнані сучасні стандарти ЕКГ, а не на довільні евристики. Це прямо вимагається розділами `Locked decisions`, `5.1 Standards anchor and evidence discipline`, `5.2 Design principles for the clinician feature space` та `Appendix A` у [AGENTS.md](/D:/GitHub/transition-matrix-ecg/AGENTS.md).

Коротко, нормативна база така:

- AHA/ACCF/HRS щодо стандартизації та інтерпретації ЕКГ, особливо частини II-VI
- ANSI/AAMI EC57 для оцінювання алгоритмів ритму та ST-сегмента, де це застосовно
- офіційна документація PTB-XL, PTB-XL+, LUDB та інших корпусів, на які посилається специфікація

Специфікація також жорстко фіксує клінічно важливі правила:

- `qtc_med_ms` використовує `QTcF`
- ST вимірюється в точці `J+60/J+80` відносно ізоелектричної лінії PR/PQ
- `rr_sdnn_ms` рахується лише за NN-інтервалами після відкидання ектопії, pacing та артефактів
- вісь QRS визначається через `atan2` з площ QRS у I та aVF
- U-хвиля дозволена лише на 500 Гц і заборонена в 100 Гц гілках
- `B_fit` будується через типізовані перетворення, а не через сирий least-squares

Отже, всі заповнені ознаки в поточних артефактах треба інтерпретувати в межах цих стандартів. Незаповнені `F24` signature scores також лишаються підпорядкованими `AGENTS.md` і мають обчислюватися лише навчальною `L1`-логістичною моделлю.

## Відмінність між `B1_raw_train`, `B1_raw_val` та `B1_raw_test`

| Матриця | Схема однакова? | Семантика розбиття | На що дозволено впливати | На що заборонено впливати |
| --- | --- | --- | --- | --- |
| `B1_raw_train.parquet` | Так, та сама 52-стовпцева схема, що й у val/test. | Зафіксована навчальна когорта PTB-XL, обмежена рівно 10 000 записами. | Перетворення тільки для навчання, майбутня підгонка композитних оцінок та оператор `A -> B_fit`, який підганяється після конвертації в `B1_fit_train`. | Не повинна містити рядки валідації/тестування або допускати витік інформації з фолду-9/фолду-10 у підігнану статистику. |
| `B1_raw_val.parquet` | Так. | Зафіксована валідаційна когорта PTB-XL з офіційного розподілу фолдів. | Вибір моделі на етапі валідації, встановлення порогів та вибір штрафу гребневої регресії після конвертації в `B1_fit_val`. | Її не можна використовувати для підгонки перетворень, головних компонент (PCA), композитних сигнатур або оператора переходу. |
| `B1_raw_test.parquet` | Так. | Остаточна відкладена тестувальна когорта PTB-XL. | Кінцева оцінка пояснюваності та перевірка лікарями після генерації прогнозів. | Повинна залишатися недоторканою під час навчання та вибору на етапі валідації. |

Усі три матриці мають однаковий порядок стовпців і семантичні визначення. Відмінність полягає не у визначенні ознак, а в належності до розбиття, кількості рядків і дозволеному статистичному використанні.

## Відмінність між `B1_*` та `B2_*`

| Аспект | `B1_*` | `B2_*` |
| --- | --- | --- |
| Джерело даних | PTB-XL v1.0.3 доповнений PTB-XL+ v1.0.1 | LUDB v1.0.1 |
| Головне призначення | Масштабна еталонна клінічна матриця | Клінічна матриця золотого стандарту розмітки |
| Основа анотації | Реальні форми хвиль плюс гармонізація PTB-XL+ та верифікація з боку форми хвиль | Реальні форми хвиль плюс ручні реперні точки LUDB як джерело істини |
| Масштаб у поточному запуску | `10000 / 2183 / 2198` рядків для train/val/test | `120 / 40 / 40` рядків у поточній виконуваній реалізації фолдів |
| Перевага | Розмір когорти та придатність для еталонного порівняння | Надійність реперних точок та точність морфології |
| Обмеження | Деякі сімейства ознак є розрідженими або залежать від якості евристичних вимірювань | Значно менший розмір когорти, тому статистична потужність нижча |
| Відповідник для вирівнювання | `A_ptbxl_train/val/test` | `A_ludb_train/val/test` |

## Технічний перелік усіх обчислених матриць

| Матриця | Джерело даних | Розбиття | Представлення | Рядки | Стовпці | Технічне значення | Parquet | CSV | Excel |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| `B1_raw_train` | PTB-XL v1.0.3 + PTB-XL+ v1.0.1 | Train | Сирий клінічний простір | 10000 | 52 | Еталонна матриця в природних одиницях, побудована з зафіксованої навчальної когорти PTB-XL у 10 000 записів. Містить 49 клінічних ознак плюс `record_id`, `qtc_formula_code` та `split`. Це авторитетна сира матриця `B`, яка використовується для підгонки перетворень лише на навчальних даних і для вирівнювання з `A_ptbxl_train` через `record_id`. | [B1_raw_train.parquet](D:/GitHub/transition-matrix-ecg/features/B1_raw_train.parquet) | [B1_raw_train.csv](D:/GitHub/transition-matrix-ecg/features/B1_raw_train.csv) | [B1_raw_train.xlsx](D:/GitHub/transition-matrix-ecg/features/B1_raw_train.xlsx) |
| `B1_raw_val` | PTB-XL v1.0.3 + PTB-XL+ v1.0.1 | Validation | Сирий клінічний простір | 2183 | 52 | Валідаційний аналог `B1_raw_train` у природних одиницях з ідентичною схемою. Він виключений з усіх етапів підгонки лише на навчальних даних і використовується винятково для вибору моделі на етапі валідації, порогів та діагностики пояснень. | [B1_raw_val.parquet](D:/GitHub/transition-matrix-ecg/features/B1_raw_val.parquet) | [B1_raw_val.csv](D:/GitHub/transition-matrix-ecg/features/B1_raw_val.csv) | [B1_raw_val.xlsx](D:/GitHub/transition-matrix-ecg/features/B1_raw_val.xlsx) |
| `B1_raw_test` | PTB-XL v1.0.3 + PTB-XL+ v1.0.1 | Test | Сирий клінічний простір | 2198 | 52 | Остаточна тестувальна матриця у природних одиницях для фолду 10 PTB-XL. Вона має той самий контракт стовпців, що й навчальна та валідаційна сирі матриці, але зарезервована для кінцевої оцінки всього пайплайну та перевірки лікарями передбачених ознак порівняно з виміряними. | [B1_raw_test.parquet](D:/GitHub/transition-matrix-ecg/features/B1_raw_test.parquet) | [B1_raw_test.csv](D:/GitHub/transition-matrix-ecg/features/B1_raw_test.csv) | [B1_raw_test.xlsx](D:/GitHub/transition-matrix-ecg/features/B1_raw_test.xlsx) |
| `B1_fit_train` | PTB-XL v1.0.3 + PTB-XL+ v1.0.1 | Train | Типізований трансформований простір підгонки | 10000 | 44 | Проєкція навчального простору `B1_raw_train`, що використовується для оцінки переходу. Безперервні ознаки вінзоризуються і піддаються z-оцінці, ознаки підрахунку використовують `log1p`, а потім z-оцінку, бінарні та обмежені ознаки використовують логіт-перетворення, `qrs_axis_deg` замінюється на `qrs_axis_sin/cos`, а стовпці без придатної навчальної статистики видаляються. | [B1_fit_train.parquet](D:/GitHub/transition-matrix-ecg/features/B1_fit_train.parquet) | [B1_fit_train.csv](D:/GitHub/transition-matrix-ecg/features/B1_fit_train.csv) | [B1_fit_train.xlsx](D:/GitHub/transition-matrix-ecg/features/B1_fit_train.xlsx) |
| `B1_fit_val` | PTB-XL v1.0.3 + PTB-XL+ v1.0.1 | Validation | Типізований трансформований простір підгонки | 2183 | 44 | Валідаційне розбиття, перетворене за допомогою пакета перетворень `B1` лише для навчальних даних. Ця матриця використовується для вибору штрафу гребневої регресії та оцінки того, наскільки добре оператор між латентним простором і лікарем узагальнює дані поза когортою підгонки. | [B1_fit_val.parquet](D:/GitHub/transition-matrix-ecg/features/B1_fit_val.parquet) | [B1_fit_val.csv](D:/GitHub/transition-matrix-ecg/features/B1_fit_val.csv) | [B1_fit_val.xlsx](D:/GitHub/transition-matrix-ecg/features/B1_fit_val.xlsx) |
| `B1_fit_test` | PTB-XL v1.0.3 + PTB-XL+ v1.0.1 | Test | Типізований трансформований простір підгонки | 2198 | 44 | Тестувальне розбиття, перетворене за допомогою того самого пакета лише для навчальних даних, що й `B1_fit_train`. Це відкладений цільовий простір для кінцевої оцінки оператора переходу та зворотного відображення у клінічні одиниці. | [B1_fit_test.parquet](D:/GitHub/transition-matrix-ecg/features/B1_fit_test.parquet) | [B1_fit_test.csv](D:/GitHub/transition-matrix-ecg/features/B1_fit_test.csv) | [B1_fit_test.xlsx](D:/GitHub/transition-matrix-ecg/features/B1_fit_test.xlsx) |
| `B2_raw_train` | LUDB v1.0.1 | Train | Сирий клінічний простір | 120 | 52 | Матриця LUDB у природних одиницях для поточного виконуваного навчального фолду. Вона має ту саму 52-стовпцеву схему, що й `B1_raw_*`, але її значення отримані з ручних реперних точок LUDB і, таким чином, забезпечують меншу, але морфологічно обґрунтовану еталонну матрицю. | [B2_raw_train.parquet](D:/GitHub/transition-matrix-ecg/features/B2_raw_train.parquet) | [B2_raw_train.csv](D:/GitHub/transition-matrix-ecg/features/B2_raw_train.csv) | [B2_raw_train.xlsx](D:/GitHub/transition-matrix-ecg/features/B2_raw_train.xlsx) |
| `B2_raw_val` | LUDB v1.0.1 | Validation | Сирий клінічний простір | 40 | 52 | Валідаційний аналог `B2_raw_train` для поточної реалізації фолдів LUDB. Він залишається поза межами статистики, що базується лише на навчальних даних, і забезпечує чисту перевірку пайплайну пояснюваності з точкою зору морфології. | [B2_raw_val.parquet](D:/GitHub/transition-matrix-ecg/features/B2_raw_val.parquet) | [B2_raw_val.csv](D:/GitHub/transition-matrix-ecg/features/B2_raw_val.csv) | [B2_raw_val.xlsx](D:/GitHub/transition-matrix-ecg/features/B2_raw_val.xlsx) |
| `B2_raw_test` | LUDB v1.0.1 | Test | Сирий клінічний простір | 40 | 52 | Остаточна відкладена матриця LUDB для поточного виконуваного фолду. Це клінічний цільовий простір, обґрунтований ручною анотацією, для перевірок позавибіркової пояснюваності на меншому наборі даних золотого стандарту. | [B2_raw_test.parquet](D:/GitHub/transition-matrix-ecg/features/B2_raw_test.parquet) | [B2_raw_test.csv](D:/GitHub/transition-matrix-ecg/features/B2_raw_test.csv) | [B2_raw_test.xlsx](D:/GitHub/transition-matrix-ecg/features/B2_raw_test.xlsx) |
| `B2_fit_train` | LUDB v1.0.1 | Train | Типізований трансформований простір підгонки | 120 | 44 | Типізоване перетворення `B2_raw_train`, створене на основі статистики LUDB лише з навчального набору. Застосовуються ті самі правила переходу «з сирого до підігнаного», що й у `B1_fit_train`, що дозволяє узгоджено формулювати оператор переходу на наборі даних золотого стандарту розмітки. | [B2_fit_train.parquet](D:/GitHub/transition-matrix-ecg/features/B2_fit_train.parquet) | [B2_fit_train.csv](D:/GitHub/transition-matrix-ecg/features/B2_fit_train.csv) | [B2_fit_train.xlsx](D:/GitHub/transition-matrix-ecg/features/B2_fit_train.xlsx) |
| `B2_fit_val` | LUDB v1.0.1 | Validation | Типізований трансформований простір підгонки | 40 | 44 | Валідаційний фолд, перетворений за допомогою навчального пакета `B2`. Він використовується для оцінки стабільності підгонки переходу для реалізації LUDB без витоку валідаційних рядків у навчальну статистику. | [B2_fit_val.parquet](D:/GitHub/transition-matrix-ecg/features/B2_fit_val.parquet) | [B2_fit_val.csv](D:/GitHub/transition-matrix-ecg/features/B2_fit_val.csv) | [B2_fit_val.xlsx](D:/GitHub/transition-matrix-ecg/features/B2_fit_val.xlsx) |
| `B2_fit_test` | LUDB v1.0.1 | Test | Типізований трансформований простір підгонки | 40 | 44 | Відкладена матриця простору підгонки LUDB, яка використовується для кінцевої оцінки золотого стандарту переходу в межах поточної експортованої реалізації фолдів. | [B2_fit_test.parquet](D:/GitHub/transition-matrix-ecg/features/B2_fit_test.parquet) | [B2_fit_test.csv](D:/GitHub/transition-matrix-ecg/features/B2_fit_test.csv) | [B2_fit_test.xlsx](D:/GitHub/transition-matrix-ecg/features/B2_fit_test.xlsx) |

## Як будуються матриці

Шлях побудови матриці складається з кількох етапів і реалізований переважно в [real_data.py](/D:/GitHub/transition-matrix-ecg/src/tm_ecg/real_data.py), [features.py](/D:/GitHub/transition-matrix-ecg/src/tm_ecg/stages/features.py) та [typed_transforms.py](/D:/GitHub/transition-matrix-ecg/src/tm_ecg/transition/typed_transforms.py).

1. `ingest --source zip` розпаковує заблоковані архіви ЕКГ у каталог [raw](/D:/GitHub/transition-matrix-ecg/raw) та записує хеші джерел у [source_manifest.json](/D:/GitHub/transition-matrix-ecg/manifests/source_manifest.json).
2. `index` створює індекси метаданих для PTB-XL та LUDB у каталозі [manifests](/D:/GitHub/transition-matrix-ecg/manifests).
3. `splits --dataset ptbxl|ludb` фіксує розподіл рядків. PTB-XL зменшується до зафіксованої навчальної когорти з `10000` рядків для `B1_train`; LUDB матеріалізується у вигляді поточної виконуваної реалізації фолдів train/val/test, що використовується для експорту `B2_*`.
4. `triads --dataset ptbxl|ludb` витягує дані вимірювань для кожного запису у каталог [interim](/D:/GitHub/transition-matrix-ecg/interim), наприклад [ptbxl_record_measurements.json](/D:/GitHub/transition-matrix-ecg/interim/ptbxl_record_measurements.json).
5. `build-b --dataset b1|b2` обчислює клінічні ознаки з Додатка А у природних одиницях та записує матриці `B_raw_*` до каталогу [features](/D:/GitHub/transition-matrix-ecg/features).
6. `fit-transition --dataset b1|b2` підганяє типізований пакет перетворень лише для навчального набору і формує матриці `B_fit_*`, що використовуються для оператора переходу гребневої регресії зі зниженим рангом.

## Неочищений простір проти простору підгонки (Raw vs Fit)

| Властивість | `B_raw_*` | `B_fit_*` |
| --- | --- | --- |
| Одиниці вимірювання | Природні клінічні одиниці, такі як `ms`, `mV`, кількості, бінарні прапорці та градуси вісі. | Значення простору моделювання після перетворень, специфічних для сімейства, та стандартизації. |
| Обробка порожніх значень (Null) | Значення зберігаються як виміряні. | Стовпці без придатної навчальної статистики видаляються; решта рядків трансформується за допомогою статистики лише з навчального набору. |
| Обробка вісі | Зберігає `qrs_axis_deg` для зручності читання лікарями. | Використовує `qrs_axis_sin` та `qrs_axis_cos` замість прямої підгонки первинного кута. |
| Основний користувач | Лікарі, пакети для огляду, безпосередня інспекція, експорт. | Підгонка оператора переходу та прогнозування в типізованому евклідовому просторі. |
| Ключ вирівнювання рядків | `record_id` | `record_id` |

## Експортовані матеріали

Кожен `.parquet` файл у каталозі [features](/D:/GitHub/transition-matrix-ecg/features) тепер експортований у два формати-близнюки:

- `.csv` для простої табличної сумісності
- `.xlsx` для перегляду на основі електронних таблиць у сумісних з Excel програмах

Файли Excel — це еквіваленти таблиць ознак Parquet на одному аркуші, які зберігають той самий порядок заголовків і вміст рядків. Файли словників уже існували у форматі CSV і не були частиною запиту на конвертацію Parquet у CSV/XLSX.

## Прямі посилання на файли

Якщо вам негайно потрібні основні матеріали, скористайтесь цими файлами:

- Сира еталонна навчальна матриця: [B1_raw_train.parquet](/D:/GitHub/transition-matrix-ecg/features/B1_raw_train.parquet), [B1_raw_train.csv](/D:/GitHub/transition-matrix-ecg/features/B1_raw_train.csv), [B1_raw_train.xlsx](/D:/GitHub/transition-matrix-ecg/features/B1_raw_train.xlsx)
- Сира еталонна валідаційна матриця: [B1_raw_val.parquet](/D:/GitHub/transition-matrix-ecg/features/B1_raw_val.parquet), [B1_raw_val.csv](/D:/GitHub/transition-matrix-ecg/features/B1_raw_val.csv), [B1_raw_val.xlsx](/D:/GitHub/transition-matrix-ecg/features/B1_raw_val.xlsx)
- Сира еталонна тестувальна матриця: [B1_raw_test.parquet](/D:/GitHub/transition-matrix-ecg/features/B1_raw_test.parquet), [B1_raw_test.csv](/D:/GitHub/transition-matrix-ecg/features/B1_raw_test.csv), [B1_raw_test.xlsx](/D:/GitHub/transition-matrix-ecg/features/B1_raw_test.xlsx)
- Сира навчальна матриця золотого стандарту: [B2_raw_train.parquet](/D:/GitHub/transition-matrix-ecg/features/B2_raw_train.parquet), [B2_raw_train.csv](/D:/GitHub/transition-matrix-ecg/features/B2_raw_train.csv), [B2_raw_train.xlsx](/D:/GitHub/transition-matrix-ecg/features/B2_raw_train.xlsx)
- Сира валідаційна матриця золотого стандарту: [B2_raw_val.parquet](/D:/GitHub/transition-matrix-ecg/features/B2_raw_val.parquet), [B2_raw_val.csv](/D:/GitHub/transition-matrix-ecg/features/B2_raw_val.csv), [B2_raw_val.xlsx](/D:/GitHub/transition-matrix-ecg/features/B2_raw_val.xlsx)
- Сира тестувальна матриця золотого стандарту: [B2_raw_test.parquet](/D:/GitHub/transition-matrix-ecg/features/B2_raw_test.parquet), [B2_raw_test.csv](/D:/GitHub/transition-matrix-ecg/features/B2_raw_test.csv), [B2_raw_test.xlsx](/D:/GitHub/transition-matrix-ecg/features/B2_raw_test.xlsx)

## Підсумки

Якщо вам потрібна основна сира еталонна матриця з вибраними клінічними ознаками у вигляді стовпців і рівно `10000` навчальними зразками, використовуйте [B1_raw_train.parquet](/D:/GitHub/transition-matrix-ecg/features/B1_raw_train.parquet) або її експортовані аналоги [B1_raw_train.csv](/D:/GitHub/transition-matrix-ecg/features/B1_raw_train.csv) та [B1_raw_train.xlsx](/D:/GitHub/transition-matrix-ecg/features/B1_raw_train.xlsx). Для отримання детальних формул по стовпцях, кількості порожніх значень та їх інтерпретації, зверніться до [b1_raw_train_null_report_uk.md](/D:/GitHub/transition-matrix-ecg/reports/b1_raw_train_null_report_uk.md).
