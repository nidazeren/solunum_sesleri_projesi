import os
import random
import numpy as np
try:
    import tensorflow as tf
except ImportError:
    tf = None
    print("UYARI: TensorFlow bulunamadı. Veri ön işleme yapılabilir ancak model eğitimi çalışmayacaktır.")

"""
Bu dosya, Solunum Sesleri Sınıflandırma (Respiratory Sound Classification)
projesinin temel yapılandırmasını içerir.

Tüm dosya yolları, model hiperparametreleri, Mel-Spektrogram dönüşüm ayarları 
ve rastgelelik (random seed) sabitleme işlemleri burada merkezi olarak yönetilir.

Bu modüler yaklaşım, projenin farklı bölümlerindeki (ön işleme, eğitim, test)
parametrelerin birbiriyle tutarlı olmasını sağlar.
"""

# ==========================================
# 1. RASTGELELİK SABİTLEME (REPRODUCIBILITY)
# ==========================================
# Akademik çalışmalarda sonuçların tekrarlanabilir (reproducible) olması çok önemlidir.
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """Tüm kütüphaneler için random seed değerini sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # GPU üzerinde deterministik (tekrarlanabilir) sonuçlar almak için:
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

set_seed()

# ==========================================
# 2. DOSYA VE DİZİN YOLLARI
# ==========================================
# Ana proje dizini (Bulunduğumuz klasör)
ANA_DIZIN = os.path.dirname(os.path.abspath(__file__))

# Veri seti yolları
VERI_SETI_DIZINI = os.path.join(ANA_DIZIN, "veri_seti")
EGITIM_VERISI_DIZINI = os.path.join(ANA_DIZIN, "egitim_verisi")
DOGRULAMA_VERISI_DIZINI = os.path.join(ANA_DIZIN, "dogrulama_verisi")
TEST_VERISI_DIZINI = os.path.join(ANA_DIZIN, "test_verisi")
SPEKTROGRAMLAR_DIZINI = os.path.join(ANA_DIZIN, "spektrogramlar")

# Model, ağırlık ve log yolları
MODELLER_DIZINI = os.path.join(ANA_DIZIN, "modeller")
AGIRLIKLAR_DIZINI = os.path.join(ANA_DIZIN, "agirliklar")
LOGLAR_DIZINI = os.path.join(ANA_DIZIN, "loglar")

# Çıktı yolları (Grafik ve Rapor)
GRAFIKLER_DIZINI = os.path.join(ANA_DIZIN, "grafikler")
RAPOR_DIZINI = os.path.join(ANA_DIZIN, "rapor")

# ==========================================
# 3. SES İŞLEME VE MEL-SPEKTROGRAM AYARLARI
# ==========================================
# Hedef örnekleme oranı (Sampling Rate - Hz). 
# Literatürde solunum sesleri için 4000 Hz ile 22050 Hz arası yaygın kullanılır.
HEDEF_SR = 22050  
HEDEF_SURE_SANIYE = 5  # Kayıtların sabitleneceği süre
MAX_SES_UZUNLUGU = HEDEF_SR * HEDEF_SURE_SANIYE

# Mel-Spektrogram Parametreleri
N_MELS = 128         # Mel frekans bandı sayısı (Y ekseni çözünürlüğü)
N_FFT = 2048         # Hızlı Fourier Dönüşümü pencere boyutu
HOP_LENGTH = 512     # Ardışık pencereler arası kaydırma miktarı
FMIN = 50            # Minimum frekans (Hz) - Düşük frekanslı gürültüleri filtrelemek için
FMAX = 4000          # Maksimum frekans (Hz) - Solunum sesleri genelde 4kHz altındadır

# Görüntü Ayarları (Model girişine uygun olacak şekilde)
GORUNTU_BOYUTU = (224, 224)  # ImageNet modelleri (MobileNet, DenseNet vb.) için standart

# ==========================================
# 4. MODEL VE EĞİTİM HİPERPARAMETRELERİ
# ==========================================
BATCH_SIZE = 32
EPOCHS_TRANSFER = 30    # Sadece sınıflandırıcı katmanların eğitileceği epoch sayısı
EPOCHS_FINE_TUNE = 20   # Tüm ağın düşük learning rate ile eğitileceği epoch sayısı
LEARNING_RATE_INITIAL = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5

# Sınıflar: Normal (0), Crackle (1), Wheeze (2), Both (3)
SINIFLAR = ["Normal", "Crackle", "Wheeze", "Both"]
NUM_CLASSES = len(SINIFLAR)
