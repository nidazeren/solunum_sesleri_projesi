import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import pandas as pd

from ayarlar import *
from model_egitimi import veri_yukleyicileri_olustur

"""
İNCE AYAR (FINE-TUNING) MODÜLÜ

Transfer Learning sonrası, modelin yeni veri setine (Solunum Sesleri) daha fazla 
adapte olabilmesi için modelin üst (son) katmanlarının kilitleri açılır.

Neden Fine-Tuning Yapıyoruz?
ImageNet ağırlıkları genel görseller (kedi, köpek, araba vb.) için optimize edilmiştir.
Mel-Spektrogram görüntüleri çok daha spesifik desenler içerir (yatay çizgiler: Wheeze, dikey çizgiler: Crackle).
Son katmanları çok düşük bir öğrenme oranıyla eğitmek, modelin spektrogramlardaki ince detayları
öğrenmesine ve doğruluğun artmasına yardımcı olur.

Kural:
- Son 20-30 katmanın kilidi açılır.
- Learning rate çok düşük tutulur (Örn: 1e-5), aksi takdirde "Catastrophic Forgetting" yaşanır ve
  önceden öğrenilmiş faydalı ağırlıklar bozulur.
"""

def ince_ayar_uygula(model_adi):
    print(f"\n{'='*50}")
    print(f"FINE-TUNING BAŞLIYOR: {model_adi}")
    print(f"{'='*50}")

    # Önceden eğitilmiş temel modeli yükle
    model_path = os.path.join(MODELLER_DIZINI, f"{model_adi}_base.h5")
    if not os.path.exists(model_path):
        print(f"HATA: {model_adi} base modeli bulunamadı! Lütfen önce model_egitimi.py çalıştırın.")
        return

    model = load_model(model_path)
    
    # Modelin tüm katmanlarını eğitime aç (kilitleri kaldır)
    for layer in model.layers:
        layer.trainable = True

    # Sadece son 30 katmanı eğitilebilir bırak, geri kalanları tekrar dondur
    FINE_TUNE_KATMAN_SAYISI = 30
    for layer in model.layers[:-FINE_TUNE_KATMAN_SAYISI]:
        layer.trainable = False

    print(f"{model_adi} modelinde toplam {len(model.layers)} katman bulunuyor.")
    print(f"Son {FINE_TUNE_KATMAN_SAYISI} katman eğitime açıldı.")

    # Çok düşük bir Learning Rate ile modeli tekrar derle
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model

def egitim_baslat():
    train_dataset, val_dataset = veri_yukleyicileri_olustur()
    modeller = ['MobileNetV2', 'DenseNet121', 'Xception']
    
    for model_adi in modeller:
        model = ince_ayar_uygula(model_adi)
        if model is None:
            continue
            
        checkpoint_path = os.path.join(MODELLER_DIZINI, f"{model_adi}_finetuned.h5")
        
        callbacks = [
            ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
            TensorBoard(log_dir=os.path.join(LOGLAR_DIZINI, f"{model_adi}_finetune"))
        ]
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS_FINE_TUNE,
            callbacks=callbacks
        )
        
        # Geçmişi kaydet
        df_history = pd.DataFrame(history.history)
        csv_path = os.path.join(LOGLAR_DIZINI, f"{model_adi}_finetune_history.csv")
        df_history.to_csv(csv_path, index=False)
        print(f"\n{model_adi} Fine-Tuning başarıyla tamamlandı ve kaydedildi.")

if __name__ == "__main__":
    egitim_baslat()
