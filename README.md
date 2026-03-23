# AI Message Fixer 

Bu proje, yapay zeka destekli bir mesaj düzenleme uygulamasıdır. Kullanıcının girdiği bozuk, günlük dildeki veya hatalı metinleri; seçilen tona göre (profesyonel, nazik, kısa vb.) yeniden kurgular.

Mimari ve Teknoloji Seçimi
Proje geliştirme sürecinde karşılaşılan API limitleri ve maliyet engellerini aşmak adına çevik bir yaklaşımla "teknoloji değişikliğine" gidilmiştir:

- **Backend:** FastAPI (Python) kullanılarak asenkron bir API yapısı kuruldu.
- **AI Motoru (LLM):** Başlangıçta planlanan OpenAI yerine, daha yüksek hız ve açık kaynak erişimi sunan **Groq Cloud** altyapısı tercih edilmiştir.
- **Model:** Meta'nın geliştirdiği **Llama-3.1-8b-instant** modeli, Groq üzerinden sisteme entegre edilmiştir.
- **Frontend:** Kullanıcı etkileşimi için Vanilla JS ve modern CSS ile duyarlı bir arayüz hazırlandı.

## Öne Çıkan Özellikler
Sıradan bir "prompt" uygulamasının ötesine geçmek için şu teknik detaylar eklenmiştir:

1. **Bilingual (Çift Dilli) Zeka:** Sistem, metnin Türkçe mi yoksa İngilizce mi olduğunu otomatik olarak algılar (`detect_language_heuristic`). Çeviri yapmadan, mesajı kendi dilinde en doğal haline getirir.
2. **Kriz Yönetimi & Adaptasyon:** Proje sırasında OpenAI kısıtlamaları, kod mimarisinde hızlı bir revizyon yapılarak Groq/Llama ekosistemine başarıyla taşınmış ve kesintisiz çalışma sağlanmıştır.
3. **Unicode & Karakter Uyumu:** Türkçe karakterlerin (ş, ğ, ç vb.) ve dilbilgisi kurallarının (ünlü uyumu gibi) LLM tarafından kusursuz uygulanması için özel prompt mühendisliği uygulanmıştır.

## Dosya Yapısı
- `backend/main.py`: Uygulamanın beyni ve AI entegrasyonu.
- `backend/requirements.txt`: Gerekli Python kütüphaneleri.
- `frontend/index.html`: Kullanıcı arayüzü.

## Hızlı Kurulum ve Çalıştırma
Projeyi yerel bilgisayarınızda test etmek için aşağıdaki adımları izleyebilirsiniz:

### 1. Gereksinimler
Bilgisayarınızda **Python 3.8+** yüklü olduğundan emin olun.

### 2. Kurulum
Proje klasörüne gidin ve gerekli kütüphaneleri yükleyin:

```bash
pip install -r backend/requirements.txt```  


### 3. API Yapılandırması
Güvenlik nedeniyle .env dosyası paylaşılmamıştır. Uygulamanın çalışması için backend/ klasörü içinde bir .env dosyası oluşturun ve ücretsiz Groq API Key anahtarınızı ekleyin:
GROQ_API_KEY=gsk_your_api_key_here

### 4. Sunucuyu Başlatma
Terminalden backend klasörüne girin ve sunucuyu çalıştırın:
cd backend
python -m uvicorn main:app --reload
Ardından frontend/index.html dosyasını tarayıcıda açarak uygulamayı kullanmaya başlayabilirsiniz.





