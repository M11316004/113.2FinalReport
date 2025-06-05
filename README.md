# 113.2FinalReport
1. 開啟 Colab 與上傳資料	
    進入 Google Colab👉 前往：https://colab.research.google.com/
   
3. Colab 上的程式碼區塊（依序執行）
butterfly_dataset 資料夾放在 Google Drive 中，路徑為： /content/drive/MyDrive/LM/Project-Butterfly/butterfly_dataset/

區塊 1：掛載 Google Drive
指令	from google.colab import drive
drive.mount('/content/drive')
結果	Mounted at /content/drive

區塊2：檔案批次處理
將檔案格式統一轉化成jpg檔案格式、檔案名稱依照img+編號.jpg方式儲存。

指令	
import os
from PIL import Image

# ✅ 原始圖片資料夾（含子資料夾）
input_root = '/content/drive/MyDrive/LM/Project-Butterfly/butterfly_dataset/train_image'

# ✅ 輸出資料夾（將重建相同子資料夾結構）
output_root = '/content/drive/MyDrive/LM/Project-Butterfly/butterfly_dataset/train_image_jpg'

# 建立輸出主資料夾
os.makedirs(output_root, exist_ok=True)

# 遍歷所有子資料夾
for subdir in os.listdir(input_root):
    subdir_path = os.path.join(input_root, subdir)
    if os.path.isdir(subdir_path):
        print(f"📁 處理資料夾: {subdir}")

        # 建立對應輸出子資料夾
        output_subdir = os.path.join(output_root, subdir)
        os.makedirs(output_subdir, exist_ok=True)

        # 收集 .jpeg / .jpg 檔案
        image_files = sorted([
            f for f in os.listdir(subdir_path)
            if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg')
        ])

        for idx, filename in enumerate(image_files, start=1):
            input_path = os.path.join(subdir_path, filename)
            output_filename = f"img{idx}.jpg"
            output_path = os.path.join(output_subdir, output_filename)

            try:
                with Image.open(input_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(output_path, format='JPEG')
                    print(f"✅ {subdir}/{output_filename}")
            except Exception as e:
                print(f"❌ 錯誤處理 {filename} in {subdir}: {e}")


區塊 3：資料擴充（Data Augmentation）
蝴蝶圖片，共六種分類，各類別約40-70 張圖。經過資料擴充，每類達200 張，共 1200 張，圖像統一調整為 150x150 像素。
指令	
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
from PIL import Image
from tqdm import tqdm

# 圖片大小與擴充目標數量
IMG_HEIGHT, IMG_WIDTH = 150, 150
TARGET_PER_CLASS = 200

# 原始與擴充後的資料夾路徑
original_data_dir = '/content/drive/MyDrive/LM/Project-Butterfly/butterfly_dataset/train_image'
augmented_data_dir = '/content/drive/MyDrive/LM/Project-Butterfly/augmented_dataset/train_image'  # ← 注意這裡直接建立在 train_image 內

# 建立資料增強器
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 逐類別執行擴充
for class_name in os.listdir(original_data_dir):
    class_path = os.path.join(original_data_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    save_class_dir = os.path.join(augmented_data_dir, class_name)
    os.makedirs(save_class_dir, exist_ok=True)

    images = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    num_existing = len(images)
    print(f"🔍 類別「{class_name}」目前有 {num_existing} 張圖片，將擴充至 {TARGET_PER_CLASS} 張")

    image_counter = 0

    # 儲存原圖（resize 後）到擴充資料夾
    for img_name in images:
        src = os.path.join(class_path, img_name)
        dst = os.path.join(save_class_dir, f"original_{img_name}")
        try:
            img = Image.open(src)
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img.save(dst)
            image_counter += 1
        except:
            print(f"⚠️ 無法處理圖片：{img_name}")

    # 執行圖片擴充直到達到目標張數
    for img_name in images:
        if image_counter >= TARGET_PER_CLASS:
            break

        img_path = os.path.join(class_path, img_name)
        try:
            img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            for batch in datagen.flow(x, batch_size=1):
                new_name = f"aug_{image_counter:04d}.jpg"
                save_path = os.path.join(save_class_dir, new_name)
                Image.fromarray((batch[0] * 255).astype(np.uint8)).save(save_path)
                image_counter += 1
                if image_counter >= TARGET_PER_CLASS:
                    break
        except:
            print(f"⚠️ 擴充失敗：{img_name}")

    print(f"✅ 類別「{class_name}」處理完成，共 {image_counter} 張圖片。\n")



區塊 4：引入必要套件並設定路徑引入必要套件並設定路徑
指令	
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# 設定路徑與參數
DATASET_PATH = '/content/drive/MyDrive/LM/Project-Butterfly/augmented_dataset/train_image'  # 替換成你實際的路徑
IMG_HEIGHT, IMG_WIDTH = 150, 150
BATCH_SIZE = 32
EPOCHS = 10

區塊 5：讀取與擴增影像資料
指令	train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

class_names = list(train_generator.class_indices.keys())
print("類別名稱：", class_names)



區塊 6：建立並訓練 CNN 模型
指令	
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)



區塊 7：儲存模型到 Google Drive
指令	
model.save('/content/drive/MyDrive/LM/Project-Butterfly/butterfly_dataset/butterfly_cnn_model.h5')


區塊 8：使用模型進行圖片預測
指令	
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ✅ 模型與基本設定
model = tf.keras.models.load_model('/content/drive/MyDrive/LM/Project-Butterfly/butterfly_dataset/butterfly_cnn_model.h5')
class_names = [
    'class1_Byasa impediens febanus_40',
    'class2_Delias pasithoe_63',
    'class3_Papilio memnon_51',
    'class4_Danaus genutia_65',
    'class5_Pieris rapae_49',
    'class6_Euploea tulliolus_66'
]
IMG_WIDTH, IMG_HEIGHT = 150, 150

# ✅ 資料夾設定
predict_folder = '/content/drive/MyDrive/LM/Project-Butterfly/butterfly_dataset/test_images'
save_folder = '/content/drive/MyDrive/LM/Project-Butterfly/prediction_results/annotated_images'
csv_path = '/content/drive/MyDrive/LM/Project-Butterfly/prediction_results/prediction_log_CNN.csv'
os.makedirs(save_folder, exist_ok=True)

# ✅ 預測與記錄容器
log_data = {
    'filename': [], 'predicted_class': [], 'true_class': [], 'confidence': [], 'match': []
}
y_true = []
y_pred = []

# ✅ 類別名稱對應表（從 class_names 抽出類別名稱）
true_labels = [name.split('_', 1)[1].rsplit('_', 1)[0] for name in class_names]

# ✅ 預測函式
def predict_images_from_folder(folder_path, model, class_names):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array, verbose=0)

        predicted_class_full = class_names[np.argmax(prediction)]
        predicted_class_name = predicted_class_full.split('_', 1)[1].rsplit('_', 1)[0]
        confidence = np.max(prediction)

        # 🔍 嘗試從檔名中找出真實類別
        matched_class = None
        for label in true_labels:
            if label.lower().replace(' ', '') in img_file.lower().replace('_', '').replace(' ', ''):
                matched_class = label
                break
        result_symbol = 'O' if matched_class == predicted_class_name else 'X'

        # ✅ 記錄資訊
        y_true.append(matched_class if matched_class else "Unknown")
        y_pred.append(predicted_class_name)

        log_data['filename'].append(img_file)
        log_data['predicted_class'].append(predicted_class_name)
        log_data['true_class'].append(matched_class if matched_class else "Unknown")
        log_data['confidence'].append(round(confidence, 4))
        log_data['match'].append(result_symbol)

        # ✅ 可視化結果儲存圖片
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image.load_img(img_path))
        ax.set_title(f'{img_file}\n預測: {predicted_class_name} ({confidence:.2f})', fontsize=9)
        ax.text(0.95, 0.05, result_symbol, transform=ax.transAxes,
                fontsize=18, fontweight='bold', color='green' if result_symbol == 'O' else 'red',
                ha='right', va='bottom')
        ax.axis('off')
        save_path = os.path.join(save_folder, f'checked_{img_file}')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        print(f"✔️ 圖片: {img_file} → 預測: {predicted_class_name}（信心值: {confidence:.2f}） → 標註：{result_symbol}")

# ✅ 執行預測
predict_images_from_folder(predict_folder, model, class_names)

# ✅ 匯出 CSV
df = pd.DataFrame(log_data)
df.to_csv(csv_path, index=False)
print(f"\n📄 預測記錄已儲存：{csv_path}")

# ✅ 混淆矩陣與報告
print("\n📊 混淆矩陣：")
labels = sorted(list(set(y_true) & set(y_pred)))
cm = confusion_matrix(y_true, y_pred, labels=labels)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
plt.title("Butterfly Classification Confusion Matrix")
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/LM/Project-Butterfly/prediction_results/confusion_matrix.png')
plt.show()

# ✅ 分類報告
print("\n📋 Classification Report:")
report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
print(report)

