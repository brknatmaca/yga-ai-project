## 🛠️ Mimari ve Teknoloji Seçimi
Proje geliştirme sürecinde karşılaşılan API limitleri ve maliyet engellerini aşmak adına çevik bir yaklaşımla "teknoloji değişikliğine" gidilmiştir:

- **Backend:** FastAPI (Python) kullanılarak asenkron bir API yapısı kuruldu.
- **AI Motoru (LLM):** Başlangıçta planlanan OpenAI yerine, daha yüksek hız ve açık kaynak erişimi sunan **Groq Cloud** altyapısı tercih edilmiştir.
- **Model:** Meta'nın geliştirdiği **Llama-3.1-8b-instant** modeli, Groq üzerinden sisteme entegre edilmiştir.
- **Frontend:** Kullanıcı etkileşimi için Vanilla JS ve modern CSS ile duyarlı bir arayüz hazırlandı.

## 🌟 Öne Çıkan Özellikler ("Kat Üstüne Kat")
Sıradan bir "prompt" uygulamasının ötesine geçmek için şu teknik detaylar eklenmiştir:

1. **Bilingual (Çift Dilli) Zeka:** Sistem, metnin Türkçe mi yoksa İngilizce mi olduğunu otomatik olarak algılar (`detect_language_heuristic`). Çeviri yapmadan, mesajı kendi dilinde en doğal haline getirir.
2. **Kriz Yönetimi & Adaptasyon:** Proje sırasında OpenAI kısıtlamaları, kod mimarisinde hızlı bir revizyon yapılarak Groq/Llama ekosistemine başarıyla taşınmış ve kesintisiz çalışma sağlanmıştır.
3. **Unicode & Karakter Uyumu:** Türkçe karakterlerin (ş, ğ, ç vb.) ve dilbilgisi kurallarının (ünlü uyumu gibi) LLM tarafından kusursuz uygulanması için özel prompt mühendisliği uygulanmıştır.
