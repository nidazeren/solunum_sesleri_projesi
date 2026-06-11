import os
import sys
import glob
import numpy as np
import librosa
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

from ayarlar import *
from mel_spektrogram_uret import gurultu_temizle, uzunluk_sabitle

"""
TAHMİN (INFERENCE) MODÜLÜ

Bu modül, dışarıdan sisteme verilen bağımsız bir `.wav` uzantılı solunum sesini alır,
aynı eğitim hattındaki gibi (gürültü temizleme -> uzunluk sabitleme -> Mel-Spektrogram)
işler ve eğitilmiş en iyi model ile tahmin (Prediction) yapar.
"""

def sesi_spektrograma_cevir(dosya_yolu):
    """Verilen ses dosyasını modelin beklediği (224, 224, 3) formatında bir spektrograma çevirir."""
    # Sesi yükle
    sinyal, _ = librosa.load(dosya_yolu, sr=HEDEF_SR)
    
    # 1. Gürültü Filtreleme (Bandpass)
    sinyal = gurultu_temizle(sinyal, HEDEF_SR)
    
    # 2. Uzunluk Sabitleme (Zero Padding veya Trimming)
    sinyal = uzunluk_sabitle(sinyal)
    
    # 3. Mel-Spektrogram Çıkarımı
    mel_spec = librosa.feature.melspectrogram(
        y=sinyal, sr=HEDEF_SR, n_fft=N_FFT, 
        hop_length=HOP_LENGTH, n_mels=N_Mels_YAPALIM_DIYECEM_AMa_AYARLARDA_VARDI_YANI_N_MELS,
        fmin=FMIN, fmax=FMAX
    )
    
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 4. Normalizasyon ve Renklendirme
    min_val = mel_spec_db.min()
    max_val = mel_spec_db.max()
    if max_val - min_val > 0:
        mel_spec_norm = 255 * (mel_spec_db - min_val) / (max_val - min_val)
    else:
        mel_spec_norm = np.zeros_like(mel_spec_db)
        
    mel_spec_norm = mel_spec_norm.astype(np.uint8)
    mel_rgb = cv2.applyColorMap(mel_spec_norm, cv2.COLORMAP_JET)
    mel_resized = cv2.resize(mel_rgb, GORUNTU_BOYUTU)
    mel_resized = cv2.flip(mel_resized, 0)
    
    # Model (1, 224, 224, 3) boyutunda bir batch bekler
    input_tensor = np.expand_dims(mel_resized, axis=0)
    
    return input_tensor

def tahmin_yap(ses_dosyasi, model_adi="DenseNet121"):
    if not os.path.exists(ses_dosyasi):
        print(f"HATA: {ses_dosyasi} bulunamadı!")
        return

    model_path = os.path.join(MODELLER_DIZINI, f"{model_adi}_finetuned.h5")
    if not os.path.exists(model_path):
        print(f"HATA: {model_adi} eğitilmiş modeli bulunamadı!")
        return
        
    print(f"\n['{ses_dosyasi}'] dosyası analiz ediliyor...")
    
    # Görüntüyü hazırla
    input_data = sesi_spektrograma_cevir(ses_dosyasi)
    
    # Modeli Yükle
    print(f"{model_adi} modeli yükleniyor...")
    model = load_model(model_path)
    
    # Tahmin
    predictions = model.predict(input_data)[0]
    
    # Sonuçları Formatla
    print("\n" + "="*40)
    print("TAHMİN SONUÇLARI")
    print("="*40)
    
    for i, sinif in enumerate(SINIFLAR):
        print(f"{sinif:<10} : % {predictions[i]*100:.2f}")
        
    en_yuksek_index = np.argmax(predictions)
    print("-" * 40)
    print(f"KARAR: Hasta büyük ihtimalle '{SINIFLAR[en_yuksek_index]}' sınıfında.")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    import ayarlar
    # ayarlar modülünde yazım hatası yaptıysam düzelteyim diye N_MELS ayarını aldım
    # Modüldeki ismi N_MELS idi
    global N_Mels_YAPALIM_DIYECEM_AMa_AYARLARDA_VARDI_YANI_N_MELS
    N_Mels_YAPALIM_DIYECEM_AMa_AYARLARDA_VARDI_YANI_N_MELS = ayarlar.N_MELS

    if len(sys.argv) > 2:
        dosya_adi = sys.argv[1]
        secilen_model = sys.argv[2]
        tahmin_yap(dosya_adi, model_adi=secilen_model)
    elif len(sys.argv) == 2:
        dosya_adi = sys.argv[1]
        print("\nBilgi: Model adı belirtilmediği için varsayılan model (DenseNet121) kullanılıyor.")
        print("Diğer modelleri test etmek için: python tahmin.py <ses_dosyasi> <ModelAdi>")
        print("Örnek: python tahmin.py test_sesleri/ornek.wav MobileNetV2\n")
        tahmin_yap(dosya_adi)
    else:
        print("Kullanım: python tahmin.py <ses_dosyasi.wav> [ModelAdi]")
        print("Model Seçenekleri: MobileNetV2, DenseNet121, Xception")
        print("Örnek bir dosya test ediliyor (varsa)...")
        # Örnek test
        ornekler = glob.glob(os.path.join(VERI_SETI_DIZINI, '**', '*.wav'), recursive=True)
        if ornekler:
            tahmin_yap(ornekler[0])
        else:
            print("Örnek ses bulunamadı.")
