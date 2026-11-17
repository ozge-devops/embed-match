import time
from sentence_transformers import SentenceTransformer, util
print("Model yükleniyor, lütfen bekleyin...")
# Hugging Face'ten kopyaladığımız model kimliği (ID) burada yer alıyor:
# 1. Modelin Yüklenmesi
print("Model yükleniyor...")
model = SentenceTransformer('intfloat/multilingual-e5-base')
# 2. Test Veri Seti (E5 modeli 'query:' ve 'passage:' formatını kullanır)
soru = "query: Sistemlerin yanıt süresi nasıl hızlandırılır?"
dokumanlar = [
    "passage: Yazılım mimarisinde mikroservisler performansı artırabilir.",
    "passage: Reducing latency in systems requires query optimization and caching.",
    "passage: Reducing latency in systems requires query optimization and caching.", # Doğru ve çok dilli yanıt
    "passage: Bugün hava güneşli ve sıcak olacak."
]
# 3. Vektörizasyon ve Performans (Hız) Ölçümü
baslangic_zamani = time.time()
soru_vektoru = model.encode(soru)
dokuman_vektorleri = model.encode(dokumanlar)
bitis_zamani = time.time()
islem_suresi = bitis_zamani - baslangic_zamani
# 4. Kosinüs Benzerliği (Cosine Similarity) Hesaplaması
skorlar = util.cos_sim(soru_vektoru, dokuman_vektorleri)[0]
print("\n--- TEST SONUÇLARI ---")
# 5. Çıktıların Raporlanması
print(f"\n--- ÇOK DİLLİ TEST SONUÇLARI ---")
print(f"Hedef Soru: {soru.replace('query: ', '')}")
for i, skor in enumerate(skorlar):
    print(f"Doküman {i+1} Skoru: {skor:.4f} | Metin: {dokumanlar[i].replace('passage: ', '')}")
    print(f"Doküman {i+1} Benzerlik Skoru: {skor:.4f} | Metin: {dokumanlar[i].replace('passage: ', '')}")
print(f"\nPerformans: Vektörizasyon işlemi {bitis_zamani - baslangic_zamani:.4f} saniye sürdü.")
print(f"\nPerformans Metriği: Vektörizasyon işlemi {islem_suresi:.4f} saniye sürmüştür.")
