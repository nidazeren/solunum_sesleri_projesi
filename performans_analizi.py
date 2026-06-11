import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ayarlar import *

"""
PERFORMANS ANALİZİ MODÜLÜ

Eğitilen modellerin Test seti üzerindeki (hiç görmedikleri hastalar) performansını ölçer.
Hesaplanan Metrikler:
- Accuracy (Doğruluk)
- Precision (Kesinlik)
- Recall (Duyarlılık)
- F1-Score
- ROC-AUC

Elde edilen sonuçlar 'rapor/model_performans_karsilastirmasi.csv' olarak kaydedilir.
"""

def test_verisini_yukle():
    print("\nTest veri seti yükleniyor...")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_VERISI_DIZINI,
        labels='inferred',
        label_mode='categorical',
        class_names=SINIFLAR,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=GORUNTU_BOYUTU,
        shuffle=False # Tahmin sırasını korumak için False olmalı!
    )
    return test_dataset

def metrikleri_hesapla(y_true, y_pred_probs):
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    try:
        # Multi-class AUC
        auc = roc_auc_score(tf.keras.utils.to_categorical(y_true, NUM_CLASSES), 
                            y_pred_probs, average='weighted', multi_class='ovr')
    except:
        auc = 0.0
        
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    }

def analiz_yap():
    test_dataset = test_verisini_yukle()
    
    # Gerçek etiketleri al
    y_true = []
    for images, labels in test_dataset:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_true = np.array(y_true)
    
    modeller = ['MobileNetV2', 'DenseNet121', 'Xception']
    sonuclar = []
    
    os.makedirs(RAPOR_DIZINI, exist_ok=True)
    rapor_dosyasi = os.path.join(RAPOR_DIZINI, "detayli_analiz_raporu.txt")
    
    with open(rapor_dosyasi, 'w', encoding='utf-8') as f:
        f.write("SOLUNUM SESLERİ - MODEL PERFORMANS ANALİZİ\n")
        f.write("="*50 + "\n\n")
        
        for model_adi in modeller:
            model_path = os.path.join(MODELLER_DIZINI, f"{model_adi}_finetuned.h5")
            if not os.path.exists(model_path):
                print(f"Uyarı: {model_adi} modeli bulunamadı, atlanıyor.")
                continue
                
            print(f"{model_adi} değerlendiriliyor...")
            model = load_model(model_path)
            
            y_pred_probs = model.predict(test_dataset)
            metrikler = metrikleri_hesapla(y_true, y_pred_probs)
            
            # Sonuçları listeye ekle
            metrikler['Model'] = model_adi
            sonuclar.append(metrikler)
            
            # Detaylı Rapor (Sklearn)
            y_pred = np.argmax(y_pred_probs, axis=1)
            report = classification_report(y_true, y_pred, target_names=SINIFLAR, zero_division=0)
            
            f.write(f"--- {model_adi} ---\n")
            f.write(report)
            f.write("\n")
            
            print(f"{model_adi} Accuracy: {metrikler['Accuracy']:.4f}")

    if sonuclar:
        df_sonuclar = pd.DataFrame(sonuclar)
        # Sütun sırasını düzenle
        df_sonuclar = df_sonuclar[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
        
        csv_yolu = os.path.join(RAPOR_DIZINI, "model_performans_karsilastirmasi.csv")
        df_sonuclar.to_csv(csv_yolu, index=False)
        
        print("\nKarşılaştırmalı Performans Tablosu:")
        print(df_sonuclar.to_string(index=False))
        print(f"\nRaporlar '{RAPOR_DIZINI}' klasörüne kaydedildi.")

if __name__ == "__main__":
    analiz_yap()
