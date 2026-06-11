import os
import glob
import pandas as pd
import numpy as np
import kagglehub
import shutil
from sklearn.model_selection import train_test_split
from ayarlar import *

"""
VERİ ÖN İŞLEME MODÜLÜ (DATA PREPROCESSING)

Bu modülün görevleri:
1. Kaggle üzerinden Respiratory Sound Database veri setini indirmek.
2. Tüm .txt dosyalarından solunum döngülerini (respiratory cycles) ve etiketlerini çıkarmak.
3. Hasta bazlı (Patient-Level) ayrım yapmak:
   - Modellerin ezberlemesini (Data Leakage) önlemek için aynı hastaya ait kayıtlar
     asla aynı anda Train ve Test setlerinde yer alamaz.
   - Hastalar %70 Eğitim (Train), %15 Doğrulama (Validation), %15 Test olarak ayrılır.
4. İşlenmiş metaveriyi (metadata) bir CSV dosyasına kaydetmek.
"""

def etiket_belirle(crackle, wheeze):
    """
    Crackle ve Wheeze değerlerine göre sınıfı belirler.
    0: Normal
    1: Crackle
    2: Wheeze
    3: Both (Crackle + Wheeze)
    """
    if crackle == 0 and wheeze == 0:
        return 0
    elif crackle == 1 and wheeze == 0:
        return 1
    elif crackle == 0 and wheeze == 1:
        return 2
    else:
        return 3

def veri_seti_indir_ve_oku():
    """
    Kaggle veri setini indirir ve metadata'yı oluşturur.
    """
    print("Kaggle'dan veri seti indiriliyor... (Bu işlem internet hızınıza göre sürebilir)")
    try:
        indirme_yolu = kagglehub.dataset_download("vbookshelf/respiratory-sound-database")
        print(f"Veri seti başarıyla indirildi: {indirme_yolu}")
    except Exception as e:
        print(f"HATA: Kaggle veri seti indirilemedi! Lütfen API anahtarınızı kontrol edin.\nDetay: {e}")
        return None

    # Ses ve metin dosyalarının olduğu dizini bulalım
    # vbookshelf/respiratory-sound-database klasör yapısında genellikle 'audio_and_txt_files' klasörü bulunur.
    hedef_dizin = None
    for root, dirs, files in os.walk(indirme_yolu):
        if any(f.endswith('.wav') for f in files):
            hedef_dizin = root
            break
            
    if hedef_dizin is None:
        print("HATA: .wav dosyaları bulunamadı!")
        return None
        
    print(f"Ses dosyaları klasörü bulundu: {hedef_dizin}")
    
    # Tüm txt dosyalarını bul
    txt_dosyaları = glob.glob(os.path.join(hedef_dizin, '*.txt'))
    
    veri_listesi = []
    
    for txt_path in txt_dosyaları:
        # Dosya adından .txt kısmını çıkarıp .wav halini bulalım
        base_name = os.path.basename(txt_path)
        wav_name = base_name.replace('.txt', '.wav')
        wav_path = os.path.join(hedef_dizin, wav_name)
        
        # Dosya isminden hasta ID'sini çıkar (İlk kısım genelde hasta ID'dir. Örn: 101_1b1_Al_sc_Meditron)
        hasta_id = int(base_name.split('_')[0])
        
        # txt dosyasını oku
        try:
            # Sütunlar: start_time, end_time, crackles, wheezes
            df_annotations = pd.read_csv(txt_path, sep='\t', header=None, 
                                         names=['start', 'end', 'crackle', 'wheeze'])
            
            for index, row in df_annotations.iterrows():
                label = etiket_belirle(row['crackle'], row['wheeze'])
                
                veri_listesi.append({
                    'hasta_id': hasta_id,
                    'dosya_yolu': wav_path,
                    'baslangic': row['start'],
                    'bitis': row['end'],
                    'crackle': row['crackle'],
                    'wheeze': row['wheeze'],
                    'sinif_id': label,
                    'sinif_adi': SINIFLAR[label]
                })
        except Exception as e:
            print(f"HATA: {txt_path} okunamadı. Detay: {e}")
            
    df = pd.DataFrame(veri_listesi)
    print(f"Toplam {len(df)} adet solunum döngüsü başarıyla çıkarıldı.")
    return df

def hasta_bazli_ayrim(df):
    """
    Veri sızıntısını (Data Leakage) önlemek amacıyla Hasta ID'sine göre veri setini
    Train (%70), Validation (%15) ve Test (%15) olarak ayırır.
    """
    print("\nHasta bazlı (Patient-level) ayırma işlemi başlatılıyor...")
    
    benzersiz_hastalar = df['hasta_id'].unique()
    np.random.shuffle(benzersiz_hastalar)
    
    # Hastaları ayırma (70-15-15)
    train_hastalar, test_val_hastalar = train_test_split(
        benzersiz_hastalar, test_size=0.30, random_state=RANDOM_SEED
    )
    
    val_hastalar, test_hastalar = train_test_split(
        test_val_hastalar, test_size=0.50, random_state=RANDOM_SEED
    )
    
    print(f"Toplam Hasta Sayısı: {len(benzersiz_hastalar)}")
    print(f"Eğitim (Train) Hasta Sayısı: {len(train_hastalar)}")
    print(f"Doğrulama (Validation) Hasta Sayısı: {len(val_hastalar)}")
    print(f"Test Hasta Sayısı: {len(test_hastalar)}")
    
    # Her hastanın hangi sette olduğunu DataFrame'e ekle
    def split_belirle(h_id):
        if h_id in train_hastalar:
            return 'train'
        elif h_id in val_hastalar:
            return 'val'
        else:
            return 'test'
            
    df['split'] = df['hasta_id'].apply(split_belirle)
    
    print("\nDöngü bazında dağılım:")
    print(df['split'].value_counts(normalize=True) * 100)
    
    return df

if __name__ == "__main__":
    print("--- VERİ ÖN İŞLEME AŞAMASI BAŞLIYOR ---")
    df_veri = veri_seti_indir_ve_oku()
    
    if df_veri is not None:
        df_veri = hasta_bazli_ayrim(df_veri)
        
        # Metadata csv dosyasını proje ana dizinine kaydet
        csv_yolu = os.path.join(ANA_DIZIN, "veri_metadata.csv")
        df_veri.to_csv(csv_yolu, index=False)
        print(f"\nMetadata dosyası kaydedildi: {csv_yolu}")
        
        print("\nSınıf Dağılımı (Tüm Veri):")
        print(df_veri['sinif_adi'].value_counts())
        
        print("\n--- VERİ ÖN İŞLEME BAŞARIYLA TAMAMLANDI ---")
        print("Sıradaki Adım: mel_spektrogram_uret.py dosyasını çalıştırarak ses dosyalarından görüntüler üretin.")
