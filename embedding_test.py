import time
from sentence_transformers import SentenceTransformer, util
print("Model yükleniyor, lütfen bekleyin...")
# Hugging Face'ten kopyaladığımız model kimliği (ID) burada yer alıyor:
model = SentenceTransformer('intfloat/multilingual-e5-base')
soru = "query: Sistemlerin yanıt süresi nasıl hızlandırılır?"
dokumanlar = [
    "passage: Yazılım mimarisinde mikroservisler performansı artırabilir.",
    "passage: Reducing latency in systems requires query optimization and caching.",
    "passage: Bugün hava güneşli ve sıcak olacak."
]
baslangic_zamani = time.time()
soru_vektoru = model.encode(soru)
dokuman_vektorleri = model.encode(dokumanlar)
bitis_zamani = time.time()
skorlar = util.cos_sim(soru_vektoru, dokuman_vektorleri)[0]
print("\n--- TEST SONUÇLARI ---")
print(f"Hedef Soru: {soru.replace('query: ', '')}")
for i, skor in enumerate(skorlar):
    print(f"Doküman {i+1} Skoru: {skor:.4f} | Metin: {dokumanlar[i].replace('passage: ', '')}")
print(f"\nPerformans: Vektörizasyon işlemi {bitis_zamani - baslangic_zamani:.4f} saniye sürdü.")
# Çok dilli anlam benzerliği
`intfloat/multilingual-e5-base` modeli ile bir soruyu belgelere karşı kosinüs benzerliğiyle sıralar. Soru Türkçe, belgelerden biri İngilizce: model çapraz dilde de yakın anlamı yüksek skorlar.
E5 ailesi kısa sorularda `query: `, belgelerde `passage: ` önekini bekler. Betik bu önekleri ekler, ekrana basarken çıkarır.
## Kurulum
GPU olmadığı için CPU tekerleği kullanılır.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
## Çalıştırma
```bash
python embedding_test.py
```
İlk çalıştırmada model Hugging Face'ten indirilir ve önbelleğe alınır. Sonraki çalıştırma
