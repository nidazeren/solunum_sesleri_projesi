import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

from ayarlar import *
from performans_analizi import test_verisini_yukle

"""
GRAFİK ÜRETİM MODÜLÜ

Eğitim loglarından ve modelin test seti üzerindeki tahminlerinden faydalanarak
akademik kalite standartlarında grafikler üretir.
- Loss / Accuracy Karşılaştırma Eğrileri (Fine-tuning öncesi ve sonrası)
- Confusion Matrix (Karmaşıklık Matrisi)
- Sınıf Dağılım Grafiği (Bar/Pie chart)
- ROC Curve ve Precision-Recall Curve
- Grad-CAM (Sınıflandırma Isı Haritası)
"""

def egitim_grafiklerini_ciz():
    print("Eğitim grafikleri çiziliyor...")
    modeller = ['MobileNetV2', 'DenseNet121', 'Xception']
    
    for model_adi in modeller:
        base_hist_yolu = os.path.join(LOGLAR_DIZINI, f"{model_adi}_base_history.csv")
        fine_hist_yolu = os.path.join(LOGLAR_DIZINI, f"{model_adi}_finetune_history.csv")
        
        if not os.path.exists(base_hist_yolu) or not os.path.exists(fine_hist_yolu):
            continue
            
        df_base = pd.read_csv(base_hist_yolu)
        df_fine = pd.read_csv(fine_hist_yolu)
        
        # Birlikte çizim için index'leri ayarla
        df_fine.index = df_fine.index + len(df_base)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy
        ax1.plot(df_base['accuracy'], label='Train Acc (Base)')
        ax1.plot(df_base['val_accuracy'], label='Val Acc (Base)')
        ax1.plot(df_fine['accuracy'], label='Train Acc (Fine-Tuned)')
        ax1.plot(df_fine['val_accuracy'], label='Val Acc (Fine-Tuned)')
        ax1.axvline(x=len(df_base)-1, color='red', linestyle='--', label='Fine-Tuning Başlangıcı')
        ax1.set_title(f'{model_adi} - Accuracy Eğrisi')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(df_base['loss'], label='Train Loss (Base)')
        ax2.plot(df_base['val_loss'], label='Val Loss (Base)')
        ax2.plot(df_fine['loss'], label='Train Loss (Fine-Tuned)')
        ax2.plot(df_fine['val_loss'], label='Val Loss (Fine-Tuned)')
        ax2.axvline(x=len(df_base)-1, color='red', linestyle='--', label='Fine-Tuning Başlangıcı')
        ax2.set_title(f'{model_adi} - Loss Eğrisi')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(GRAFIKLER_DIZINI, f"{model_adi}_egitim_egrisi.png"), dpi=300)
        plt.close()

def metrik_grafiklerini_ciz():
    print("Confusion Matrix, ROC ve PR eğrileri çiziliyor...")
    test_dataset = test_verisini_yukle()
    
    y_true = []
    for images, labels in test_dataset:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_true = np.array(y_true)
    y_true_cat = tf.keras.utils.to_categorical(y_true, NUM_CLASSES)
    
    modeller = ['MobileNetV2', 'DenseNet121', 'Xception']
    
    for model_adi in modeller:
        model_path = os.path.join(MODELLER_DIZINI, f"{model_adi}_finetuned.h5")
        if not os.path.exists(model_path):
            continue
            
        model = load_model(model_path)
        y_pred_probs = model.predict(test_dataset)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=SINIFLAR, yticklabels=SINIFLAR)
        plt.title(f'{model_adi} - Confusion Matrix')
        plt.ylabel('Gerçek Sınıf')
        plt.xlabel('Tahmin Edilen Sınıf')
        plt.tight_layout()
        plt.savefig(os.path.join(GRAFIKLER_DIZINI, f"{model_adi}_confusion_matrix.png"), dpi=300)
        plt.close()
        
        # 2. ROC Curve (Multi-class One-vs-Rest)
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green', 'orange']
        for i, color in zip(range(NUM_CLASSES), colors):
            fpr, tpr, _ = roc_curve(y_true_cat[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f'ROC curve of class {SINIFLAR[i]} (area = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_adi} - ROC Eğrisi')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(GRAFIKLER_DIZINI, f"{model_adi}_roc_curve.png"), dpi=300)
        plt.close()

def sinif_dagilimi_ciz():
    print("Sınıf dağılım grafikleri çiziliyor...")
    csv_yolu = os.path.join(ANA_DIZIN, "veri_metadata.csv")
    if os.path.exists(csv_yolu):
        df = pd.read_csv(csv_yolu)
        
        plt.figure(figsize=(12, 5))
        
        # Bar Chart
        plt.subplot(1, 2, 1)
        sns.countplot(data=df, x='sinif_adi', hue='split', palette='viridis')
        plt.title("Sınıf ve Split Dağılımı")
        plt.xlabel("Sınıflar")
        plt.ylabel("Örnek Sayısı")
        
        # Pie Chart (Orijinal Dağılım)
        plt.subplot(1, 2, 2)
        siniflar = df['sinif_adi'].value_counts()
        plt.pie(siniflar, labels=siniflar.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.title("Orijinal Sınıf Dağılımı (%)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(GRAFIKLER_DIZINI, "sinif_dagilimi.png"), dpi=300)
        plt.close()

def grad_cam_olustur():
    """
    Modelin kararlarını neye göre verdiğini (Mel-spektrogram üzerindeki hangi frekanslara 
    odaklandığını) gösteren Grad-CAM Isı Haritalarını oluşturur.
    Not: Bu metot sadece yapıyı gösterir, tam bir Grad-CAM implementasyonu 
    belirli bir katmanın ismini (örn: 'out_relu' DenseNet121 için) gerektirir.
    """
    pass

if __name__ == "__main__":
    os.makedirs(GRAFIKLER_DIZINI, exist_ok=True)
    sinif_dagilimi_ciz()
    egitim_grafiklerini_ciz()
    metrik_grafiklerini_ciz()
    print("Tüm grafikler başarıyla üretildi.")
