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
İlk çalıştırmada model Hugging Face'ten indirilir ve önbelleğe alınır. Sonraki çalıştırmalar yalnızca vektörleştirmeyi ölçer.
