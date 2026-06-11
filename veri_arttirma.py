import os
import pandas as pd
import numpy as np
import librosa
import random
from tqdm import tqdm
from ayarlar import *
from mel_spektrogram_uret import gurultu_temizle, uzunluk_sabitle, spektrogram_olustur_ve_kaydet

"""
VERİ ARTIRMA (DATA AUGMENTATION) MODÜLÜ

Respiratory Sound Database (Solunum Sesleri Veri Seti) ciddi bir sınıf dengesizliğine sahiptir.
'Normal' sınıfı çok fazla örneğe sahipken, 'Wheeze' ve 'Both' (Crackle+Wheeze) sınıfları azınlıktadır.
Modelin çoğunluk sınıfına "ezberlemesini" (overfitting) engellemek için, azınlık sınıflarına
sentetik veri artırma yöntemleri uygulanarak sınıflar dengelenir.

Kullanılan Yöntemler (Neden kullanıldıkları):
1. Time Stretching (Zaman Uzatma/Kısaltma): Nefes alıp verme hızı insandan insana değişebilir.
2. Pitch Shifting (Ses Tonu Değiştirme): Akciğer kapasitesi ve fizyolojik farklılıklar frekansı etkiler.
3. Gaussian Noise (Gürültü Ekleme): Hastane ortamı, steteskop sürtünmesi gibi gerçek dünya gürültülerini simüle eder.
"""

def add_gaussian_noise(sinyal, noise_factor=0.005):
    """Sinyale rastgele Gaussian gürültü ekler."""
    noise = np.random.randn(len(sinyal))
    augmented_data = sinyal + noise_factor * noise
    return augmented_data

def time_stretch(sinyal, rate=1.2):
    """Sinyalin hızını değiştirir (Tonu bozmadan)."""
    return librosa.effects.time_stretch(y=sinyal, rate=rate)

def pitch_shift(sinyal, sr, n_steps=2):
    """Sinyalin tonunu (frekansını) kaydırır (Hızı bozmadan)."""
    return librosa.effects.pitch_shift(y=sinyal, sr=sr, n_steps=n_steps)

def augmentasyon_uygula(sinyal):
    """
    Rastgele bir augmentasyon yöntemi seçip uygular.
    Her yöntem solunum sesinin doğal yapısına (Crackle, Wheeze) farklı bir varyasyon katar.
    """
    secim = random.choice([1, 2, 3])
    
    if secim == 1:
        # Hız değiştir (0.8 ile 1.2 arası rastgele bir oran)
        rate = random.uniform(0.8, 1.2)
        aug_sinyal = time_stretch(sinyal, rate)
    elif secim == 2:
        # Ton değiştir (-2 ile 2 adım arası rastgele)
        steps = random.randint(-2, 2)
        aug_sinyal = pitch_shift(sinyal, HEDEF_SR, steps)
    else:
        # Gürültü ekle
        factor = random.uniform(0.001, 0.01)
        aug_sinyal = add_gaussian_noise(sinyal, factor)
        
    return aug_sinyal

def veri_dengeleme_islemi():
    csv_yolu = os.path.join(ANA_DIZIN, "veri_metadata.csv")
    if not os.path.exists(csv_yolu):
        print("HATA: veri_metadata.csv bulunamadı!")
        return
        
    df = pd.read_csv(csv_yolu)
    
    # Sadece eğitim verisinde augmentasyon yapılmalıdır! Validation ve Test orijinalliğini korumalıdır.
    df_train = df[df['split'] == 'train']
    
    # Sınıf sayılarını bul
    sinif_sayilari = df_train['sinif_adi'].value_counts()
    print("Eğitim Seti Orijinal Sınıf Dağılımı:")
    print(sinif_sayilari)
    
    max_ornek_sayisi = sinif_sayilari.max()
    
    print(f"\nHer sınıf {max_ornek_sayisi} örneğe tamamlanacak (Oversampling via Augmentation).")
    
    for sinif_adi in SINIFLAR:
        mevcut_sayi = sinif_sayilari.get(sinif_adi, 0)
        eksik_sayi = max_ornek_sayisi - mevcut_sayi
        
        if eksik_sayi <= 0:
            continue # Çoğunluk sınıfını atla
            
        print(f"\n[{sinif_adi}] sınıfı için {eksik_sayi} adet sentetik veri üretilecek...")
        
        # Bu sınıfa ait tüm eğitim dosyaları
        sinif_df = df_train[df_train['sinif_adi'] == sinif_adi]
        
        for i in tqdm(range(eksik_sayi), desc=f"{sinif_adi} Sentetik Üretim"):
            # Rastgele bir orijinal kayıt seç
            ornek_row = sinif_df.sample(1).iloc[0]
            
            try:
                # Sesi yükle
                dosya_yolu = ornek_row['dosya_yolu']
                baslangic = ornek_row['baslangic']
                sure = ornek_row['bitis'] - ornek_row['baslangic']
                sinyal, _ = librosa.load(dosya_yolu, sr=HEDEF_SR, offset=baslangic, duration=sure)
                
                # Gürültü filtrele
                sinyal = gurultu_temizle(sinyal, HEDEF_SR)
                
                # Augmentasyon uygula
                aug_sinyal = augmentasyon_uygula(sinyal)
                
                # Uzunluğu eşitle (Time stretch nedeniyle süre değişmiş olabilir)
                aug_sinyal = uzunluk_sabitle(aug_sinyal)
                
                # Kayıt için benzersiz isim (aug_...)
                resim_adi = f"aug_{ornek_row['hasta_id']}_{ornek_row.name}_{i}.png"
                kayit_yolu = os.path.join(EGITIM_VERISI_DIZINI, sinif_adi, resim_adi)
                
                # Spektrograma çevir ve kaydet
                spektrogram_olustur_ve_kaydet(aug_sinyal, kayit_yolu)
                
            except Exception as e:
                pass
                
    print("\n--- VERİ ARTIRMA VE DENGELEME İŞLEMİ TAMAMLANDI ---")

if __name__ == "__main__":
    veri_dengeleme_islemi()
