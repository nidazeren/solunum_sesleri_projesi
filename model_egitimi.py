import os
import ssl
# Mac cihazlarda indirme sırasında yaşanan SSL sertifika hatasını (CERTIFICATE_VERIFY_FAILED) çözer
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, DenseNet121, Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from tensorflow.keras.optimizers import Adam
import pandas as pd

from ayarlar import *

"""
MODEL EĞİTİMİ (TRANSFER LEARNING) MODÜLÜ

Bu modülde önceden eğitilmiş (Pre-Trained) ImageNet ağırlıklarına sahip
derin öğrenme modelleri kullanılarak Solunum Sesi sınıflandırması yapılır.

Kullanılan Modeller:
1. MobileNetV2: Düşük parametre sayısı, hızlı ve mobil/edge cihazlara uygun.
2. DenseNet121: Yoğun bağlantılar sayesinde özellikleri kaybetmeden derin katmanlara aktarır (Özellikle Wheeze gibi ince frekansları yakalamada etkilidir).
3. Xception: Depthwise Separable Convolution kullanarak geniş özellik çıkarımı yapar.

İlk aşamada (Transfer Learning):
- Base modelin ağırlıkları dondurulur (trainable = False).
- Sadece eklenen son sınıflandırıcı katmanlar eğitilir.
- Overfitting'i önlemek için Dropout, BatchNormalization ve EarlyStopping kullanılır.
"""

def veri_yukleyicileri_olustur():
    """Klasörlerden eğitim, doğrulama ve test veri setlerini yükler."""
    
    print("\nEğitim veri seti yükleniyor...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        EGITIM_VERISI_DIZINI,
        labels='inferred',
        label_mode='categorical',
        class_names=SINIFLAR,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=GORUNTU_BOYUTU,
        shuffle=True,
        seed=RANDOM_SEED
    )

    print("\nDoğrulama (Validation) veri seti yükleniyor...")
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        DOGRULAMA_VERISI_DIZINI,
        labels='inferred',
        label_mode='categorical',
        class_names=SINIFLAR,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=GORUNTU_BOYUTU,
        shuffle=False,
        seed=RANDOM_SEED
    )
    
    # Performans artırımı için veri önyükleme (Prefetching)
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_dataset, val_dataset

def model_olustur(model_adi):
    """
    Belirtilen önceden eğitilmiş modeli dondurup,
    yeni bir sınıflandırıcı kafa (classification head) ekleyerek döndürür.
    """
    input_shape = GORUNTU_BOYUTU + (3,)
    
    # 1. Base Modeli Seç ve Yükle
    if model_adi == 'MobileNetV2':
        base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif model_adi == 'DenseNet121':
        base_model = DenseNet121(input_shape=input_shape, include_top=False, weights='imagenet')
    elif model_adi == 'Xception':
        base_model = Xception(input_shape=input_shape, include_top=False, weights='imagenet')
    else:
        raise ValueError("Geçersiz model adı!")

    # 2. Base Modeli Dondur (Ağırlıklar güncellenmeyecek)
    base_model.trainable = False
    
    # 3. Yeni Sınıflandırıcı Katmanlar (Classification Head)
    x = base_model.output
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    
    x = BatchNormalization()(x)
    x = Dropout(0.5, name="dropout_1")(x) # Overfitting önleme
    
    x = Dense(256, activation='relu', name="dense_256")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3, name="dropout_2")(x)
    
    outputs = Dense(NUM_CLASSES, activation='softmax', name="predictions")(x)
    
    model = Model(inputs=base_model.input, outputs=outputs, name=f"{model_adi}_Transfer")
    
    # 4. Modeli Derle
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_INITIAL),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def egitim_baslat():
    train_dataset, val_dataset = veri_yukleyicileri_olustur()
    modeller = ['MobileNetV2', 'DenseNet121', 'Xception']
    
    for model_adi in modeller:
        print(f"\n{'='*50}")
        print(f"EĞİTİM BAŞLIYOR: {model_adi} (Transfer Learning)")
        print(f"{'='*50}")
        
        tamamlandi_dosyasi = os.path.join(LOGLAR_DIZINI, f"{model_adi}_tamamlandi.txt")
        csv_path = os.path.join(LOGLAR_DIZINI, f"{model_adi}_base_history.csv")
        checkpoint_path = os.path.join(MODELLER_DIZINI, f"{model_adi}_base.h5")
        latest_checkpoint_path = os.path.join(MODELLER_DIZINI, f"{model_adi}_latest.h5")
        
        # 1. Model zaten tamamlandıysa atla
        if os.path.exists(tamamlandi_dosyasi):
            print(f"[{model_adi}] modeli zaten başarıyla eğitilmiş. Atlanıyor...")
            continue
            
        if os.path.exists(csv_path) and not os.path.exists(latest_checkpoint_path):
            print(f"[{model_adi}] modeli zaten başarıyla eğitilmiş (Eski formata göre). Atlanıyor...")
            with open(tamamlandi_dosyasi, "w") as f:
                f.write("Tamamlandi.")
            continue

        # 2. Modeli oluştur veya son durumdan yükle
        initial_epoch = 0
        if os.path.exists(latest_checkpoint_path):
            print(f"Kaldığı yerden devam ediliyor: {latest_checkpoint_path} yükleniyor...")
            model = load_model(latest_checkpoint_path)
            if os.path.exists(csv_path):
                # Kaçıncı epoch'ta kaldığını bul
                df = pd.read_csv(csv_path)
                initial_epoch = len(df)
                print(f"Kaldığı Epoch: {initial_epoch}")
        else:
            print(f"Yeni model oluşturuluyor: {model_adi}")
            model = model_olustur(model_adi)
            # Eğer önceki yarım kalan history veya base checkpoint varsa temizle ki temiz başlasın
            if os.path.exists(csv_path):
                os.remove(csv_path)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
        
        # 3. Callback'ler
        callbacks = [
            # En iyi ağırlıkları kaydet
            ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
            # Her epoch sonunda en son durumu kaydet (kaldığı yerden devam edebilmek için)
            ModelCheckpoint(filepath=latest_checkpoint_path, monitor='val_loss', save_best_only=False, verbose=0),
            # Gelişme durduğunda eğitimi kes
            EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
            # Plato'ya ulaşıldığında Learning Rate'i düşür
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
            # TensorBoard ile görselleştirme logları
            TensorBoard(log_dir=os.path.join(LOGLAR_DIZINI, f"{model_adi}_base")),
            # Her epoch history'sini csv'ye anında yaz
            CSVLogger(csv_path, append=True)
        ]
        
        if initial_epoch >= EPOCHS_TRANSFER:
            print(f"[{model_adi}] eğitimi tamamlanmış görünüyor (Epoch sınırı aşıldı).")
            with open(tamamlandi_dosyasi, "w") as f:
                f.write("Tamamlandi.")
            continue
        
        # Eğitimi Başlat
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS_TRANSFER,
            initial_epoch=initial_epoch,
            callbacks=callbacks
        )
        
        print(f"\n{model_adi} modeli başarıyla eğitildi ve {checkpoint_path} yoluna kaydedildi.")
        
        # Eğitim başarıyla bittiği için latest checkpoint'i sil ve flag koy
        if os.path.exists(latest_checkpoint_path):
            os.remove(latest_checkpoint_path)
        with open(tamamlandi_dosyasi, "w") as f:
            f.write("Tamamlandi.")

if __name__ == "__main__":
    os.makedirs(MODELLER_DIZINI, exist_ok=True)
    os.makedirs(LOGLAR_DIZINI, exist_ok=True)
    egitim_baslat()
