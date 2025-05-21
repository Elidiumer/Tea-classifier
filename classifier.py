import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageOps, UnidentifiedImageError
from customtkinter import CTkImage
from ultralytics import YOLO
import time
from enum import Enum
import os

class LogLevel(
    Enum
):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


LOG_LEVEL = LogLevel.DEBUG

LOG_COLORS = {
    LogLevel.DEBUG: "\033[94m",
    LogLevel.INFO: "\033[92m",
    LogLevel.WARNING: "\033[93m",
    LogLevel.ERROR: "\033[91m",
}

RESET_COLOR = "\033[0m"

def log(
        level,
        message
):

    if level.value >= LOG_LEVEL.value:

        timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        color = LOG_COLORS.get(
            level,
            ""
        )

        print(
            f"{color}[{timestamp}] [{level.name}] {message}{RESET_COLOR}"
        )

def log_debug(
        message
):
    log(
        LogLevel.DEBUG,
        message
    )

def log_info(
        message
):
    log(
        LogLevel.INFO,
        message
    )

def log_warning(
        message
):
    log(
        LogLevel.WARNING,
        message
    )

def log_error(
        message
):
    log(
        LogLevel.ERROR,
        message
    )

def print_supported_image_formats():

    formats = [
        "*.jpg",
        "*.jpeg",
        "*.png",
    ]

    log_info(
        "Поддерживаемые форматы изображений:"
    )

    for fmt in formats:

        log_debug(
            f" - {fmt}"
        )

def check_model_files():

    missing = False

    if not os.path.exists(
            "best.pt"
    ):

        log_warning(
            "Файл модели 'best.pt' не найден!"
        )

        missing = True

    if not os.path.exists(
            "best_fermented.pt"
    ):

        log_warning(
            "Файл модели 'best_fermented.pt' не найден!"
                    )

        missing = True

    if not missing:

        log_info(
            "Оба файла модели найдены."
        )

def print_available_tea_classes():
    log_info(
        "Доступные классы чая:"
    )
    for idx, name in enumerate(
            tea_classes
    ):
        log_debug(
            f"{idx}: {name}"
        )

def print_available_fermentation_classes():
    log_info(
        "Доступные классы ферментации:"
    )
    for idx, name in enumerate(
            fermentation_classes
    ):
        log_debug(
            f"{idx}: {name}"
        )


def analyze_project_structure(
        root="."
):
    log_info(
        f"Анализ структуры проекта в '{os.path.abspath(root)}'..."
    )

    for dirpath, dirnames, filenames in os.walk(
            root
    ):
        depth = dirpath.count(
            os.sep
        ) - root.count(
            os.sep
        )

        indent = "  " * depth

        log_debug(f"{indent}[Папка] {os.path.basename(dirpath)} ({len(filenames)} файлов)")

        for filename in filenames:
            full_path = os.path.join(
                dirpath,
                filename
            )
            try:
                size_kb = os.path.getsize(
                    full_path
                ) / 1024

                log_debug(f"{indent}  └─ {filename} — {size_kb:.1f} КБ")

            except Exception as e:
                log_warning(
                    f"Не удалось получить размер файла: {full_path}. Ошибка: {e}"
                )

def check_model_sizes():
    log_info(
        "Проверка размеров файлов моделей..."
    )

    model_files = {
        "best.pt": "Модель классификации чая",
        "best_fermented.pt": "Модель классификации ферментации"
    }

    for filename, description in model_files.items():
        if os.path.exists(
                filename
        ):

            size_mb = os.path.getsize(
                filename
            ) / (
                    1024 ** 2
            )

            log_debug(
                f"{description} ({filename}) — {size_mb:.2f} МБ"
            )

        else:
            log_warning(
                f"{description} ({filename}) не найден."
            )


def scan_image_folder(
        folder_path="images"
):
    log_info(
        f"Поиск изображений в папке: {folder_path}"
    )

    if not os.path.exists(
            folder_path
    ):
        log_warning(
            f"Папка не найдена: {folder_path}"
        )
        return

    valid_extensions = (
        ".jpg", ".jpeg", ".png"
    )

    count = 0
    for fname in os.listdir(
            folder_path
    ):
        if fname.lower().endswith(
                valid_extensions
        ):
            full_path = os.path.join(
                folder_path,
                fname
            )
            try:
                with Image.open(
                        full_path
                ) as img:
                    img.verify()
                log_debug(f"Изображение проверено: {fname}")
                count += 1
            except (
                    UnidentifiedImageError,
                    IOError
            ):
                log_warning(
                    f"Поврежденное изображение: {fname}"
                )

    log_info(
        f"Проверено изображений: {count}"
    )


log_info(
    "Инициализация классов чая..."
)

tea_classes = [
    'Красный чай',
    'Улун',
    'Белый чай',
    'Желтый чай',
    'Зеленый чай',
    'Шу пуэр',
    'Шен пуэр'
]

log_info(
    "Инициализация классов ферментации..."
)

fermentation_classes = [
    'Слабоферментированный чай',
    'Полуферментированный чай',
    'Ферментированный чай',
    'Постферментированный чай'
]

log_info(
    "Загрузка модели детекции..."
)


model_detect = YOLO(
    "best.pt"
)


log_info(
    "Модель детекции загружена"
)


log_info(
    "Загрузка модели классификации..."
)


model_classify = YOLO(
    "best_fermented.pt"
)


log_info(
    "Модель классификации загружена"
)


log_info(
    "Настройка интерфейса..."
)


ctk.set_appearance_mode(
    "System"
)


ctk.set_default_color_theme(
    "blue"
)


app = ctk.CTk()


app.title(
    "Классификатор сырья чая"
)


app.geometry(
    "800x600"
)


image_label = ctk.CTkLabel(
    app,
    text=""
)


image_label.pack(
    pady=10
)


result_box = ctk.CTkTextbox(
    app,
    height=200,
    width=600
)


result_box.pack(
    pady=10
)

image_path = ""

image_prepared = None

check_model_sizes()

print_available_tea_classes()

print_available_fermentation_classes()


def prepare_image_512(
        pil_img
):
    width, height = pil_img.size

    print(
        f"[INFO] Оригинальный размер изображения: {width}x{height}"
    )

    if width == 512 and height == 512:

        print(
            "[INFO] Размер изображения уже 512x512, подготовка не требуется."
        )

        return pil_img

    if width >= 512 and height >= 512:
        left = (
                       width - 512
               ) // 2

        top = (
                      height - 512
              ) // 2

        right = left + 512

        bottom = top + 512

        print(
            "[INFO] Изображение обрезано до 512x512."
        )

        return pil_img.crop(
            (
                left,
                top,
                right,
                bottom
            )
        )

    delta_w = max(
        0,
        512 - width
    )

    delta_h = max(0,
                  512 - height
                  )

    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - delta_w // 2,
        delta_h - delta_h // 2
    )

    padded_img = ImageOps.expand(
        pil_img,
        padding,
        fill=(
            0,
            0,
            0
        )
    )

    print(
        "[INFO] Изображение дополнено до 512x512."
    )

    return padded_img.resize(
        (
            512,
            512
        )
    )


def select_image():

    global image_path, image_prepared

    file_path = filedialog.askopenfilename(
        filetypes=[
            (
                "Image files",
                "*.jpg *.png *.jpeg"
            )
        ]
    )

    if file_path:

        image_path = file_path

        print(
            f"[INFO] Изображение выбрано: {file_path}"
        )


        img_original = Image.open(
            file_path
        ).convert(
            "RGB"
        )

        image_prepared = prepare_image_512(
            img_original
        )


        preview = CTkImage(
            light_image=img_original,
            dark_image=img_original,
            size=(
                400, 300
            )
        )

        image_label.configure(
            image=preview,
            text=""
        )

        image_label.image = preview

        result_box.delete(
            "0.0",
            "end"
        )

        result_box.insert(
            "0.0",
            "Изображение загружено.\nНажмите 'Анализировать' для предсказания.\n"
        )


def analyze():

    if not image_path or image_prepared is None:

        print(
            "[WARN] Попытка анализа без изображения."
              )

        result_box.insert(
            "0.0",
            "Сначала выберите изображение.\n"
        )

        return

    result_box.delete(
        "0.0",
        "end"
    )

    print(
        "[INFO] Запуск модели детекции..."
          )

    result_box.insert(
        "end",
        "Детекция объектов (класс):\n"
    )

    results1 = model_detect(
        image_prepared
    )

    best_box = None

    best_conf = 0.0

    for result in results1:

        for box in result.boxes:

            conf = float(
                box.conf[
                    0
                ]
            )

            print(
                f"[DEBUG] Найден класс: {int(box.cls[0])}, уверенность: {conf:.2f}"
            )

            if conf > best_conf:

                best_conf = conf

                best_box = box

    cls_id = int(
        best_box.cls
        [
            0
        ]
    )

    class_name = tea_classes[
        cls_id
    ] if cls_id < len(
        tea_classes
    ) else f"Класс {cls_id}"

    print(
        f"[INFO] Лучший класс: {class_name},"
        f" уверенность: {best_conf:.2f}"
    )

    result_box.insert(
        "end",
        f"Объект — {class_name},"
        f" уверенность: {best_conf:.2f}\n"
    )

    print(
        "[INFO] Запуск модели классификации ферментации..."
    )

    result_box.insert(
        "end",
        "\nПроверка второй моделью (ферментация):\n"
    )

    results2 = model_classify(
        image_prepared
    )

    best_box2 = None

    best_conf2 = 0.0

    for result in results2:

        for box in result.boxes:

            conf = float(
                box.conf[
                    0
                ]
            )

            print(
                f"[DEBUG] Найден класс ферментации: {int(box.cls[0])},"
                f" уверенность: {conf:.2f}"
            )

            if conf > best_conf2:

                best_conf2 = conf

                best_box2 = box

    cls_id2 = int(
        best_box2.cls[
            0
        ]
    )

    ferment_name = fermentation_classes[
        cls_id2
    ] if cls_id2 < len(
        fermentation_classes
    ) else f"Класс {cls_id2}"

    print(
        f"[INFO] Лучший ферментационный класс: {ferment_name},"
        f" уверенность: {best_conf2:.2f}"
    )

    result_box.insert(
        "end",
        f"Ферментация — {ferment_name},"
        f" уверенность: {best_conf2:.2f}\n"
    )

    print(
        "[INFO] Анализ завершён."
    )


btn_frame = ctk.CTkFrame(
    app
)

btn_frame.pack(
    pady=10
)

load_btn = ctk.CTkButton(
    btn_frame,
    text="Выбрать изображение",
    command=select_image
)

load_btn.pack(
    side="left",
    padx=10
)

run_btn = ctk.CTkButton(
    btn_frame,
    text="Анализировать",
    command=analyze
)

run_btn.pack(
    side="left",
    padx=10
)


print(
    "[INFO] Приложение запущено."
)

app.mainloop()
