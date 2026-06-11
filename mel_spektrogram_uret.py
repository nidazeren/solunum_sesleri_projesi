import os
import pandas as pd
import numpy as np
import librosa
import cv2
import scipy.signal
from tqdm import tqdm
from ayarlar import *

"""
MEL-SPEKTROGRAM ÜRETİM MODÜLÜ

Bu modülün görevleri:
1. `veri_onisleme.py` tarafından üretilen 'veri_metadata.csv' dosyasını okumak.
2. Her bir solunum döngüsü için belirtilen başlangıç ve bitiş sürelerine göre
   orijinal `.wav` dosyasından ilgili kesiti almak.
3. Ses sinyali üzerinde gürültü temizleme (Bandpass filter 50Hz-4000Hz) uygulamak.
4. Ses uzunluklarını sabitlemek (Zero padding veya Trimming).
5. Sinyali Normalize etmek.
6. Librosa ile Mel-Spektrogram üretip, desibel (dB) ölçeğine çevirmek.
7. Spektrogramı 0-255 piksel aralığına normalize edip (224x224) RGB PNG olarak kaydetmek.
8. Kaydedilen görüntüleri Train/Val/Test klasörlerine ve alt sınıf klasörlerine dağıtmak.
"""

def klasorleri_hazirla():
    """Keras image_dataset_from_directory için uygun klasör ağacını oluşturur."""
    print("Veri dizinleri ve alt sınıf klasörleri hazırlanıyor...")
    split_dizinler = {
        'train': EGITIM_VERISI_DIZINI,
        'val': DOGRULAMA_VERISI_DIZINI,
        'test': TEST_VERISI_DIZINI
    }
    
    for _, dizin in split_dizinler.items():
        for sinif_adi in SINIFLAR:
            yol = os.path.join(dizin, sinif_adi)
            os.makedirs(yol, exist_ok=True)
            
    return split_dizinler

def gurultu_temizle(sinyal, sr):
    """
    Solunum sesleri genellikle 50Hz ile 4000Hz arasındadır.
    Bu bandın dışında kalan düşük frekanslı (kalp atışı, hareket) ve 
    yüksek frekanslı (mikrofon hışırtısı vb.) gürültüleri filtreler.
    """
    nyquist = 0.5 * sr
    low = FMIN / nyquist
    high = FMAX / nyquist
    # Butterworth bandpass filtresi
    b, a = scipy.signal.butter(4, [low, high], btype='band')
    temiz_sinyal = scipy.signal.filtfilt(b, a, sinyal)
    return temiz_sinyal

def uzunluk_sabitle(sinyal):
    """
    Tüm sesleri sabit uzunluğa getirir. 
    Kısaysa sonuna sıfır ekler, uzunsa keser.
    """
    if len(sinyal) > MAX_SES_UZUNLUGU:
        # Ortadan kesmeyi tercih edebiliriz, ama genelde baştan almak standarttır
        return sinyal[:MAX_SES_UZUNLUGU]
    elif len(sinyal) < MAX_SES_UZUNLUGU:
        pad_length = MAX_SES_UZUNLUGU - len(sinyal)
        # Sinyalin iki tarafına eşit sıfır ekle
        return np.pad(sinyal, (0, pad_length), mode='constant')
    return sinyal

def spektrogram_olustur_ve_kaydet(sinyal, kayit_yolu):
    """
    Sinyali Mel-Spektrogram resmine (PNG) dönüştürür ve kaydeder.
    MobileNetV2, DenseNet121 gibi ImageNet modelleri 3 kanallı (RGB) ve genelde 224x224 resim bekler.
    """
    # 1. Mel-spektrogram hesapla
    mel_spec = librosa.feature.melspectrogram(
        y=sinyal, sr=HEDEF_SR, n_fft=N_FFT, 
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX
    )
    
    # 2. Amplitüd değerini Desibele (dB) çevir
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 3. Piksel değerlerini 0-255 arasına (Min-Max) normalize et
    min_val = mel_spec_db.min()
    max_val = mel_spec_db.max()
    if max_val - min_val > 0:
        mel_spec_norm = 255 * (mel_spec_db - min_val) / (max_val - min_val)
    else:
        mel_spec_norm = np.zeros_like(mel_spec_db)
        
    mel_spec_norm = mel_spec_norm.astype(np.uint8)
    
    # Keras Modelleri 3 boyutlu resim beklediği için gri formatı renkli haritaya çevir (COLORMAP)
    # COLORMAP_JET yaygın kullanılır, spektrogram detaylarını iyi yansıtır.
    mel_rgb = cv2.applyColorMap(mel_spec_norm, cv2.COLORMAP_JET)
    
    # Resmi yeniden boyutlandır (224x224)
    mel_resized = cv2.resize(mel_rgb, GORUNTU_BOYUTU)
    
    # OpenCV, Y-eksenini ters gösterir, Spektrogramda düşük frekanslar altta olmalıdır.
    mel_resized = cv2.flip(mel_resized, 0)
    
    # PNG olarak kaydet
    cv2.imwrite(kayit_yolu, mel_resized)

def islem_baslat():
    csv_yolu = os.path.join(ANA_DIZIN, "veri_metadata.csv")
    if not os.path.exists(csv_yolu):
        print("HATA: veri_metadata.csv bulunamadı! Önce veri_onisleme.py dosyasını çalıştırın.")
        return
        
    df = pd.read_csv(csv_yolu)
    split_dizinler = klasorleri_hazirla()
    
    print(f"\nToplam işlenecek kayıt sayısı: {len(df)}")
    print("Mel-Spektrogram dönüşümü başlıyor...")
    
    basarili = 0
    hatali = 0
    
    # tqdm ile progress bar oluştur
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Spektrogramlar Üretiliyor"):
        try:
            dosya_yolu = row['dosya_yolu']
            baslangic = row['baslangic']
            bitis = row['bitis']
            split_turu = row['split']
            sinif_adi = row['sinif_adi']
            hasta_id = row['hasta_id']
            
            # Sadece ilgili kısmı okumak (offset ve duration ile)
            sure = bitis - baslangic
            sinyal, sr = librosa.load(dosya_yolu, sr=HEDEF_SR, offset=baslangic, duration=sure)
            
            # Gürültü filtreleme
            sinyal = gurultu_temizle(sinyal, HEDEF_SR)
            
            # Uzunluk eşitleme
            sinyal = uzunluk_sabitle(sinyal)
            
            # Kayıt yolu (Benzersiz isim: split_hastaID_index.png)
            resim_adi = f"{split_turu}_{hasta_id}_{index}.png"
            kayit_dizini = os.path.join(split_dizinler[split_turu], sinif_adi)
            kayit_yolu = os.path.join(kayit_dizini, resim_adi)
            
            # Üret ve kaydet
            spektrogram_olustur_ve_kaydet(sinyal, kayit_yolu)
            
            basarili += 1
            
        except Exception as e:
            hatali += 1
            # Çok fazla hata logu ekrana basmamak için sessiz geçiyoruz. İstenirse eklenebilir.
            pass
            
    print("\n--- İŞLEM TAMAMLANDI ---")
    print(f"Başarılı Dönüşüm: {basarili}")
    print(f"Hatalı/Atlanan Dönüşüm: {hatali}")

if __name__ == "__main__":
    islem_baslat()
