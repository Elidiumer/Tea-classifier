import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageOps
from customtkinter import CTkImage
from ultralytics import YOLO



tea_classes = [
    'Красный чай',
    'Улун',
    'Белый чай',
    'Желтый чай',
    'Зеленый чай',
    'Шу пуэр',
    'Шен пуэр'
]


fermentation_classes = [
    'Слабоферментированный чай',
    'Полуферментированный чай',
    'Ферментированный чай',
    'Постферментированный чай'
]


model_detect = YOLO("best.pt")
model_classify = YOLO("best_fermented.pt")


ctk.set_appearance_mode("System")  # "Light", "Dark", "System"
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Классификатор сырья чая")
app.geometry("800x600")

image_label = ctk.CTkLabel(app, text="")
image_label.pack(pady=10)

result_box = ctk.CTkTextbox(app, height=200, width=600)
result_box.pack(pady=10)

image_path = ""
image_prepared = None



def prepare_image_512(pil_img):
    width, height = pil_img.size

    if width == 512 and height == 512:
        return pil_img

    if width >= 512 and height >= 512:
        left = (width - 512) // 2
        top = (height - 512) // 2
        right = left + 512
        bottom = top + 512
        return pil_img.crop((left, top, right, bottom))

    delta_w = max(0, 512 - width)
    delta_h = max(0, 512 - height)
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    padded_img = ImageOps.expand(pil_img, padding, fill=(0, 0, 0))
    return padded_img.resize((512, 512))



def select_image():
    global image_path, image_prepared
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        image_path = file_path
        img_original = Image.open(file_path).convert("RGB")
        image_prepared = prepare_image_512(img_original)

        preview = CTkImage(light_image=img_original, dark_image=img_original, size=(400, 300))
        image_label.configure(image=preview, text="")
        image_label.image = preview

        result_box.delete("0.0", "end")
        result_box.insert("0.0", "Изображение загружено.\nНажмите 'Анализировать' для предсказания.\n")



def analyze():
    if not image_path or image_prepared is None:
        result_box.insert("0.0", "Сначала выберите изображение.\n")
        return

    result_box.delete("0.0", "end")


    result_box.insert("end", "Детекция объектов (класс):\n")
    results1 = model_detect(image_prepared)

    best_box = None
    best_conf = 0.0

    for result in results1:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box

    cls_id = int(best_box.cls[0])
    class_name = tea_classes[cls_id] if cls_id < len(tea_classes) else f"Класс {cls_id}"
    result_box.insert("end", f"Объект — {class_name}, уверенность: {best_conf:.2f}\n")


    result_box.insert("end", "\nПроверка второй моделью (ферментация):\n")
    results2 = model_classify(image_prepared)

    best_box2 = None
    best_conf2 = 0.0

    for result in results2:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > best_conf2:
                best_conf2 = conf
                best_box2 = box

    cls_id2 = int(best_box2.cls[0])
    ferment_name = fermentation_classes[cls_id2] if cls_id2 < len(fermentation_classes) else f"Класс {cls_id2}"
    result_box.insert("end", f"Ферментация — {ferment_name}, уверенность: {best_conf2:.2f}\n")




btn_frame = ctk.CTkFrame(app)
btn_frame.pack(pady=10)

load_btn = ctk.CTkButton(btn_frame, text="Выбрать изображение", command=select_image)
load_btn.pack(side="left", padx=10)

run_btn = ctk.CTkButton(btn_frame, text="Анализировать", command=analyze)
run_btn.pack(side="left", padx=10)


app.mainloop()
