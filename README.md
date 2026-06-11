# Solunum Sesleri Sınıflandırma Analizi: Transfer Learning Yaklaşımları Üzerine Karşılaştırmalı Bir Çalışma

## 1. Giriş
Bu proje, ICBHI Respiratory Sound Database veri seti kullanılarak solunum seslerindeki patolojik belirtilerin (Crackle ve Wheeze) derin öğrenme ve sinyal işleme yöntemleriyle tespit edilmesini amaçlamaktadır. Ses sinyalleri öncelikle spektro-temporal analiz teknikleri (Mel-Spektrogram) ile iki boyutlu görsel özellik haritalarına dönüştürülmüş, ardından önceden eğitilmiş (Pre-Trained) konvolüsyonel sinir ağları (MobileNetV2, DenseNet121, Xception) kullanılarak sınıflandırma modelleri geliştirilmiştir. 

Projenin temel motivasyonu; solunum yolu hastalıklarının erken teşhisinde, veri sızıntısını (data leakage) önleyen titiz bir değerlendirme metodolojisi sunmak ve sınıf dengesizliklerini sentetik veri üretimi (Data Augmentation) ile aşarak objektif, tekrarlanabilir bir sistem tasarlamaktır.

## 2. Veri Ön İşleme ve Metodoloji

### 2.1. Hasta Bazlı (Patient-Level) Çapraz Doğrulama
Solunum sesleri veri setlerindeki en büyük risk faktörü "Veri Sızıntısı" (Data Leakage) problemidir. Rastgele veri ayrımı yapılması durumunda aynı hastaya ait farklı solunum döngüleri hem eğitim hem de test setine düşebilir. Bu durum modelin, hastanın patolojisini öğrenmek yerine hastanın "ses tonunu" ezberlemesine yol açar. Bu projede tam hasta bazlı (Strict Patient-Level) izolasyon sağlanmış; %70 Eğitim, %15 Doğrulama, %15 Test ayrımı tamamen benzersiz hastalar üzerinden gerçekleştirilmiştir.

### 2.2. Sinyal İşleme (Mel-Spektrogram) ve Gürültü Filtreleme
Solunum sesleri büyük oranda 50Hz ile 4000Hz arasında yoğunlaşır. `scipy` tabanlı Butterworth Bandpass filtresi kullanılarak kalp atışları gibi düşük frekanslı gürültüler (50Hz altı) ve ortam hışırtıları (4000Hz üstü) temizlenmiştir. Temizlenen sinyaller `librosa` yardımıyla Mel-Spektrogram'lara dönüştürülmüş, logaritmik desibel (dB) ölçeğine geçilerek düşük amplitüdlü fakat karakteristik solunum patolojileri vurgulanmıştır.

### 2.3. Sınıf Dengesizliği ve Veri Artırma (Data Augmentation)
Veri setinde "Normal" sınıfı domine ederken, patolojik sınıflar ("Wheeze", "Both") ciddi oranda azınlıktadır. Bu durum modelin sürekli çoğunluk sınıfını tahmin ederek (Accuracy paradoksu) yanıltıcı yüksek başarılar göstermesine sebep olur. Bu sorunu aşmak için eğitim setindeki azınlık sınıflarına spesifik ses artırma yöntemleri uygulanmıştır:
*   **Time Stretching:** Farklı nefes alış-veriş ritimlerini simüle etmek.
*   **Pitch Shifting:** Göğüs kafesi yapısına bağlı akustik rezonans farklılıklarını taklit etmek.
*   **Gaussian Noise:** Hastane/Klinik ortam gürültülerini (steteskop sürtünmesi vb.) modelleyerek ağın gürültüye dayanıklılığını (robustness) artırmak.

## 3. Deneysel Mimariler ve Mühendislik Analizi

Transfer learning aşamasında üç farklı mimari kıyaslanmıştır:

### 3.1. DenseNet121
*   **Mühendislik Yorumu:** Klasik CNN mimarilerinde derinlik arttıkça kaybolan mikro özellikler, DenseNet121’in yoğun bağlantılı (Dense Blocks) yapısı sayesinde sonraki katmanlara gradyan bozulması yaşanmadan aktarılabilmiştir. Bu durum özellikle *ince wheeze (hırıltı)* frekans desenlerinin korunmasına büyük katkı sağlamıştır. Her katmanın önceki tüm katmanlardan gelen özellikleri birleştirerek işlemesi (feature reuse), mel-spektrogramlardaki karmaşık, birbirinin içine geçmiş frekans yapılarını ayrıştırmada en yüksek başarıyı göstermesini sağlamıştır.

### 3.2. Xception
*   **Mühendislik Yorumu:** Depthwise Separable Convolution mekanizmasını kullanan Xception, kanal bazlı (spatial) özellikleri çapraz kanal (cross-channel) ilişkilerinden ayırarak öğrenir. Bu mekanizma, spektrogramlardaki yatay çizgiler (wheeze - sürekli frekans) ve dikey bantlar (crackle - ani patlamalar) arasındaki ilişkileri izole etmekte teorik olarak oldukça güçlüdür. Ancak yüksek parametre sayısına sahip olması, nispeten küçük olan bu veri setinde (Data Augmentation'a rağmen) overfitting (aşırı öğrenme) riskini artırmış, DenseNet kadar stabil bir doğrulama performansı sergileyememiştir.

### 3.3. MobileNetV2
*   **Mühendislik Yorumu:** Inverted Residuals ve Linear Bottlenecks yapısıyla tasarlanan MobileNetV2, çok düşük işlemci maliyeti ve bellek tüketimiyle öne çıkar. Darboğaz (bottleneck) yapısı, solunum seslerindeki en kritik özellikleri sıkıştırarak çıkarılmasını zorlasa da, Crackle (çıtırtı) gibi ani spektral değişimlerde bilgi kaybı (information loss) yaşatabilmektedir. Mobil sağlık (mHealth) uygulamalarında, örneğin dijital bir steteskop uygulamasında gerçek zamanlı çıkarım (real-time inference) yapabilmek için ideal bir mimari olmasına karşın doğruluk oranı bakımından daha derin ağların (DenseNet) gerisinde kalmıştır.

## 4. Fine-Tuning Stratejisi
Ağlar sadece son sınıflandırma kafa katmanı ile eğitildikten sonra "Catastrophic Forgetting" (Öğrenilmiş özelliklerin yok olması) problemini engellemek adına son 30 katmanın kilitleri açılmış ve *Learning Rate* `1e-5` gibi çok düşük bir değere çekilmiştir (ReduceLROnPlateau destekli). Bu ince ayar (Fine-Tuning), modellerin genel ImageNet özelliklerinden sıyrılıp, görsel olarak çizgisel bir karakter sergileyen "solunum spektrogramlarına" özel filtreler geliştirmesini sağlamıştır. Eğrilerde (Loss ve Accuracy) fine-tuning başlangıcı (dikey kırmızı çizgi sonrası) ani bir uyumlanma gözlemlenmiştir.

## 5. Sonuç ve Öneriler
*   Model değerlendirmesi salt `Accuracy` metrisiyle değil, imbalans veri setlerinde çok daha gerçekçi bir metrik olan `ROC-AUC` ve `F1-Score` üzerinden yapılmıştır.
*   En kararlı performansı DenseNet121 sergilemiştir.
*   Gelecek çalışmalarda, temporal dinamikleri daha iyi yakalamak için CNN mimarilerinin sonuna LSTM veya GRU gibi tekrarlayan (Recurrent) katmanlar eklenebilir veya Vision Transformers (ViT) tabanlı bir yaklaşım denenebilir.
