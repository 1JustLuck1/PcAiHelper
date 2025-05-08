document.addEventListener('DOMContentLoaded', function() {
    Vue.createApp({
        delimiters: ['[[', ']]'],
        data() {
            return {
                currentQuestion: 1,
                selectedAnswers: [],
                questions: [
                    {
                        id: 1,
                        text: "Для чего вы будете использовать ПК?",
                        options: [
                            { id: "3d", text: "3D-рендер и анимация" },
                            { id: "video", text: "Видеомонтаж" },
                            { id: "gaming", text: "Гейминг" },
                            { id: "ml", text: "Машинное обучение" }
                            // { id: "office", text: "Офисная работа" }
                        ],
                        hint: "Сфера деятельности, для которой вы планируете собирать конфигурацию."
                    },
                    {
                        id: 2,
                        text: "Какой у вас бюджет?",
                        options: [
                            { id: "min", text: "Минимальный" },
                            { id: "base", text: "Базовый" },
                            { id: "medium", text: "Средний" },
                            { id: "max", text: "Максимальный" }
                        ],
                        hint: "Бюжет определяет, какие компоненты вы можете приобрести.\n<b>Минимальный</b> до ~100тыс. рублей,\n<b>Базовый</b> ~110-140тыс.рублей,\n<b>Средний</b> ~150-200тыс.руб,\n<b>Максимальный</b> ~250тыс.руб и более"
                    },
                    {
                        id: 3,
                        text: "Уровень решаемых задач в вашей деятельности?",
                        options: [
                            { id: "fullhd", text: "Базовый" },
                            { id: "quadhd", text: "Продвинутый/Полупрофессиональный" },
                            { id: "ultrahd", text: "Профессиональный" }
                        ],
                        hint: "<b>Базовый уровень</b> - самый распространенный, покрывает большество задач обычного пользователя(работа, игры, мультимедия).\n<b>Продвинутый/Полу-профессиональный</b> - уровень задач требующих больше мощностей, в виду специфики рабочей деятельности.\n<b>Профессиональный</b> - задачи максимальной сложности, которые требуют максимальной производительности."
                    },
                    {
                        id: 4,
                        text: "Желаемый вендор платформы?",
                        options: [
                            { id: "amd", text: "AMD" },
                            { id: "intel", text: "Intel" }
                        ],
                        hint: "Ведор платформы, определеяется производителем центрального процессора или сокетом материнской платы"
                    },
                    {
                        id: 5,
                        text: "Желаемый вендор видеокарты?",
                        options: [
                            { id: "nvidia", text: "NVIDIA" },
                            { id: "amd", text: "AMD" },
                            { id: "intel", text: "Intel" }
                        ],
                        hint: "Ведор платформы, определеяется производителем графического чипа видеокарты"
                    },
                    {
                        id: 6,
                        text: "Нужен ли небольшой запас на будущее?",
                        options: [
                            { id: "0", text: "Запас не требуется" },
                            { id: "1", text: "Запас на будущее нужен" }
                        ],
                        hint: "Подразумавается, что при выборе запаса на будущее, пользователь получит сборку мощнее на 10-15% по синтетическим показателям PassMark (https://www.passmark.com/)."
                    },
                    {
                        id: 7,
                        text: "Какой тип видеокарты требуется?",
                        options: [
                            { id: "uni", text: "Универсальная" },
                            { id: "professonal", text: "Профессиональная" }
                            // { id: "office", text: "Офисная" }
                        ],
                        hint: "Универсальная подойдет для задач широкого профиля. \nПрофессиональная создана специально для вычислиния множества парралельных опрераций в сферах, где это требуется. \nОфисная это доступный вариант для обычных рабочих задач и работы в нетребовальтехных приложениях."
                    }
                ],
                activeHint: "",
                cpu: "",
                gpu: "",
                ram: "",
                psu: "",
                psu_vendor: "",
                mb_vendor: "",
                cpu_cooler_vendor: "",
                cpudata: [],
                gpudata: [],
                cpu_links: [],
                gpu_links: [],
                psu_links: [],
                mb_links: [],
                cpu_cooler_links: [],
                loading: false,
                error: null
            }
        },
        created() {
            this.selectedAnswers = Array(this.questions.length).fill(null);
        },
        computed: {
            isFormComplete() {
                return this.selectedAnswers.every(answer => answer !== null);
            }
        },
        methods: {
            nextQuestion() {
                if (this.currentQuestion < this.questions.length) {
                    this.currentQuestion++;
                }
            },
            prevQuestion() {
                if (this.currentQuestion > 1) {
                    this.currentQuestion--;
                }
            },
            showHint(hintText) {
                this.activeHint = hintText;
            },
            formatContent(text) {
                return text.replace(/\n/g, '<br>');
            },
            submitSurvey() {
                const surveyData = {
                    answers: this.selectedAnswers.map((answer, index) => ({
                        questionId: this.questions[index].id,
                        answerId: answer
                    }))
                };

                fetch("/api/configure", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(surveyData),
                })
                .then((response) => {
                    if (!response.ok) throw new Error('Ошибка обработки моделью');
                    return response.json();
                })
                .then((data) => {
                    console.log("Data received:", data);
                    this.cpu = data.cpu;
                    this.gpu = data.gpu;
                    this.ram = data.ram;
                    this.cpudata = data.cpud;
                    this.gpudata = data.gpud;
                    this.psu = data.total_tdp;
                    this.psu_vendor = data.psu_vendor;
                    this.mb_vendor = data.mb_vendor;
                    this.cpu_cooler_vendor = data.cpu_cooler_vendor;
                    this.cpu_links = data.cpu_links;
                    this.gpu_links = data.gpu_links;
                    this.psu_links = data.psu_links;
                    this.mb_links = data.mb_links;
                    this.cpu_cooler_links = data.cpu_cooler_links;
                })
                .catch(error => {
                    this.error = error.message;
                    console.error('Ошибка', error);
                });
            },
            downloadExcel() {
                // Подготавливаем только данные компонентов
                const exportData = {
                    cpu: this.cpu,
                    gpu: this.gpu,
                    ram: this.ram,
                    psu: this.psu,
                    psu_vendor: this.psu_vendor,
                    mb_vendor: this.mb_vendor,
                    cpu_cooler_vendor: this.cpu_cooler_vendor
                };
        
                fetch("/download-excel", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(exportData),
                })
                .then(response => {
                    if (!response.ok) throw new Error('Ошибка генерации Excel');
                    return response.blob();
                })
                .then(blob => {
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'pc_components.xlsx';
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Произошла ошибка при генерации Excel файла');
                });
            }
        }
    }).mount('#configuration');
    
    Vue.createApp({
        delimiters: ['[[', ']]'],
        data() {
            return{
                cpus_options: [],
                selectedCpu: "",
                gpus_options: [],
                selectedGpu: "",
                selectedSphere: 'gaming',
                selectedLvl: 'base',
                cpuScore: null,
                gpuScore: null,
                cpuDescription: "",
                gpuDescription: "",
                loading: false,
                error: null
            }
        },
        mounted() {
            this.fetchOptions();
        },
        methods: {
            fetchOptions() {
                this.loading = true;
                this.error = null;
                
                fetch("/api/cpus")
                    .then(response => {
                        if (!response.ok) throw new Error('Ошибка загрузки CPU');
                        return response.json();
                    })
                    .then(data => {
                        this.cpus_options = data;
                    })
                    .catch(error => {
                        this.error = error.message;
                        console.error('CPU load error:', error);
                    });
                fetch("/api/gpus")
                    .then(response => {
                    if (!response.ok) throw new Error('Ошибка загрузки GPU');
                    return response.json();
                    })
                    .then(data => {
                    this.gpus_options = data;
                    })
                    .catch(error => {
                    this.error = this.error ? this.error + ' | ' + error.message : error.message;
                    console.error('GPU load error:', error);
                    })
                    .finally(() => {
                    this.loading = false;
                });
                
            },
            getRatingClass(score) {
                if (score >= 120) return 'excellent';
                if (score >= 100) return 'good';
                if (score >= 80) return 'medium';
                if (score >= 50) return 'bad';
                return 'poor';
            },
            submitForm() {
                this.loading = true;
                this.error = false;
                
                // Отправка данных на сервер Flask
                fetch('api/evaluate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        cpu: this.selectedCpu,
                        gpu: this.selectedGpu,
                        sphere: this.selectedSphere,
                        lvl: this.selectedLvl
                    })
                })
                .then(response => response.json())
                .then(data => {
                    this.loading = false;
                    this.cpuScore = data.cpu_percent;
                    this.gpuScore = data.gpu_percent;
                    this.cpuDescription = data.cpu_quote;
                    this.gpuDescription = data.gpu_quote
                })
                .catch(error => {
                    this.error = true;
                    console.error('Error:', error);
                });
            }
        }
    }).mount('#evaluate')

    Vue.createApp({
        delimiters: ['[[', ']]'],
        data() {
            return{
                items: [
                    {
                        title: 'Гайд по сборке ПК для 3D Рендера и Анимации',
                        content: `<b>Центральный процессор (ЦП)</b>
                        Раньше ЦП был ключевым для рендеринга, теперь видеокарты также важны, но процессор остаётся критичным для общей производительности. Современные ЦП используют многоядерные архитектуры. Intel и AMD сейчас предлагают конкурентоспособные решения. При выборе учитывайте совместимость с материнской платой, ОЗУ и охлаждением.

                        <b>Видеокарта (GPU)</b>
                        В рендеринге GPU важнее, чем в других задачах, и работает значительно быстрее CPU. Лучший выбор — NVIDIA из-за CUDA, которая эффективнее OpenCL (AMD). Если приложение поддерживает CUDA, предпочтительнее NVIDIA.

                        <b>Оперативная память (ОЗУ)</b>
                        ОЗУ не главный ограничивающий фактор. Минимум — 8 ГБ, лучше больше для 3D-моделирования. DDR5 в двух- или четырёхканальной конфигурации даёт ощутимый прирост.

                        <b>Накопители</b>

                            - HDD: 1–2 ТБ хватает для 3D-дизайна, но анимация требует больше.
                            - SSD: Ускоряют систему. SATA SSD — оптимальный выбор, NVMe (PCIe) даёт прирост только при работе с большими файлами.

                        <b>Блок питания (БП)</b>
                        Важен для стабильности системы. Выбирайте проверенные модели с хорошими отзывами. Качественный БП прослужит 5–10 лет.`,
                        isOpen: false
                    },
                    {
                        title: 'Гайд по сборке ПК для Видеомонтажа',
                        content: `<b>Центральный процессор (ЦП)</b>
                        Центральный процессор — ключевой компонент для видеомонтажа. Чем больше ядер и выше тактовая частота, тем быстрее работает ПК. Современные редакторы (Premiere Pro, Final Cut Pro) активно используют многоядерные CPU. Для серьёзной работы нужен минимум 6-ядерный процессор. В DaVinci Resolve видеокарта важнее, но CPU остаётся критичным.
                        
                        <b>Видеокарта (GPU)</b>
                        В большинстве видеоредакторов GPU менее важен, чем CPU. Исключение — DaVinci Resolve, где рендеринг идёт на видеокарте. NVIDIA предпочтительнее из-за CUDA, но AMD с OpenCL тоже подходит. Для игр GPU важнее.
                        
                        <b>Оперативная память (ОЗУ)</b>

                            - 8 ГБ для 1080p,
                            - 16 ГБ для 4K.
                            - 32+ ГБ для сложных проектов.

                        <b>Накопители</b>

                            - SSD — быстрее, особенно для ОС, программ и активных проектов.
                            - HDD — дешевле, подходит для хранения архивов.
                            
                            Оптимальная схема:

                            - SSD под систему и софт,
                            - Отдельный SSD под проекты,
                            - HDD для хранения.

                        <b>Материнская плата</b>
                        Главное — совместимость с CPU, ОЗУ и SSD. Дополнительно:

                            - USB 3.0+,
                            - Достаточно SATA-портов,
                            - Хороший звук (или отдельная звуковая карта).

                        <b>Блок питания (БП)</b>
                        Выбирайте надёжный БП (80+ Bronze или выше). Слабый блок может повредить компоненты.`,
                        isOpen: false
                    },
                    {
                        title: 'Гайд по сборке ПК для Игр',
                        content: `<b>Процессор (CPU)</b>
                        Игры зависят от CPU, но не так сильно, как от видеокарты. Для комфортного геймплея хватит 6-ядерного процессора (например, Intel Core i5 или AMD Ryzen 5). Если бюджет позволяет, можно взять 8-ядерный (i7/Ryzen 7) — это полезно для стриминга и игр с открытым миром.

                        <b>Видеокарта (GPU)</b>
                        Главный компонент игрового ПК. Чем мощнее GPU, тем выше FPS и детализация. Для Full HD хватит NVIDIA RTX 3060 / AMD RX 6600, для 1440p — RTX 4070 / RX 7800 XT, для 4K — RTX 4080 / RX 7900 XTX. NVIDIA лучше для трассировки лучей и DLSS, AMD — для соотношения цена/производительность.

                        <b>Оперативная память (RAM)</b>
                        Минимум — 16 ГБ DDR4/DDR5 (для современных AAA-игр). Для стриминга и мультизадачности лучше 32 ГБ. Частота 3200–3600 МГц (для AMD Ryzen важнее).

                        <b>Накопители</b>
                            - SSD NVMe (1 ТБ) — для системы и игр (быстрая загрузка уровней).
                            - HDD (2+ ТБ) — для хранения медиафайлов и старых игр (если нужен большой объём).

                        <b>Материнская плата</b>
                        Следует выбирать под сокет CPU (LGA 1700 для Intel, AM4/AM5 для AMD). Важно:
                            - PCIe 4.0/5.0 для видеокарты и SSD,
                            - 2+ слота M.2,
                            - Хороший VRM (если планируется разгон).

                        <b>Блок питания (PSU)</b>
                        Лучше брать с запасом мощности:
                            - 650–750 Вт для средних сборок,
                            - 850+ Вт для топовых GPU (RTX 4090, RX 7900 XTX).
                            - Обязательно 80+ Bronze/Gold и проверенный бренд (например, Seasonic, Corsair, EVGA).

                        <b>Охлаждение</b>
                            - Боксовый кулер — для базовых сборок.
                            - Башня/СЖО — для разгона и мощных CPU.
                            - Корпус с хорошей вентиляцией (3+ вентилятора, mesh-передняя панель).`,
                        isOpen: false
                    },
                    {
                        title: 'Гайд по сборке ПК для Машинного обучения',
                        content: `<b>Процессор (CPU)</b>
                        Не главный компонент, но важен для препроцессинга и небольших моделей. Лучше брать многоядерные CPU (AMD Ryzen 9 / Threadripper или Intel Core i9) — они ускоряют обработку данных.

                        <b>Видеокарта (GPU)</b>
                        Основа для обучения нейросетей. NVIDIA предпочтительнее из-за CUDA и поддержки фреймворков (TensorFlow, PyTorch). Оптимальные варианты:
                            - RTX 4090 (24 ГБ) — лучший выбор,
                            - RTX 3090 / 4080 (24/16 ГБ) — чуть медленнее,
                            - RTX A6000 (48 ГБ) — для серьёзных задач, но дорого.
                            - AMD хуже из-за слабой поддержки в ML-библиотеках.

                        <b>Оперативная память (RAM)</b>
                        Чем больше, тем лучше:
                            - 32 ГБ — минимум для средних моделей,
                            - 64–128 ГБ — для работы с большими датасетами.
                            - Лучше DDR4/DDR5 с высокой частотой (3600+ МГц).

                        <b>Накопители</b>
                            - Быстрый NVMe SSD (1–2 ТБ) — для ОС, кода и временных данных,
                            - Второй SSD или HDD (4+ ТБ) — для хранения датасетов.

                        <b>Материнская плата</b>
                        Главное — поддержка:
                            - Много RAM (4+ слота),
                            - Несколько PCIe x16 (если планируется несколько GPU),
                            - M.2 для NVMe.
                            - Для Threadripper/Xeon нужны платы с TRX4/LGA 4677.

                        <b>Блок питания (PSU)</b>
                            - 850–1000 Вт для одной топовой GPU,
                            - 1200+ Вт для нескольких видеокарт.
                            - Только 80+ Platinum/Titanium (Corsair AX, Seasonic PRIME).

                        <b>Охлаждение</b>
                            - Процессор — мощный кулер или СЖО,
                            - Видеокарта — хорошая вентиляция в корпусе (желательно с доп. вентиляторами),
                            - Корпус — Full-Tower для лучшего airflow и места под GPU.`,
                        isOpen: false
                    }
                ]
            };
        },
        methods: {
            formatContent(text) {
                return text.replace(/\n/g, '<br>');
            },
            toggleItem(index) {
                this.items[index].isOpen = !this.items[index].isOpen;
            }
        }
    }).mount('#info');
    
});