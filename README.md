# Çok dilli anlam benzerliği
`intfloat/multilingual-e5-base` modeli ile bir soruyu belgelere karşı kosinüs benzerliğiyle sıralar. Soru Türkçe, belgelerden biri İngilizce: model çapraz dilde de yakın anlamı yüksek skorlar.
`intfloat/multilingual-e5-base` modeli bir soru ile üç belgeyi vektöre çevirir ve kosinüs benzerliğiyle sıralar. Soru Türkçedir. Belgelerden biri İngilizcedir; model aynı anlama gelen metni dil değişse de yukarı taşır.
E5 ailesi kısa sorularda `query: `, belgelerde `passage: ` önekini bekler. Betik bu önekleri ekler, ekrana basarken çıkarır.
E5 ailesi sorunun başına `query: `, belgenin başına `passage: ` koyulmasını ister. `embedding_test.py` bu önekleri metnin içine yazar, ekrana basarken çıkarır.
Türkçe bir soruyu üç belgeyle karşılaştırdım. Önce `intfloat/multilingual-e5-base` ile baktım. Sonra aynı cümleleri MiniLM ve `bge-m3` ile de geçirdim.
## Model karşılaştırması
| Model Adı | Parametre / Boyut | Lisans | Performans ve Yük |
| --- | --- | --- | --- |
| intfloat/multilingual-e5-base | 278M (1,11 GB) | MIT | 768 boyut, en fazla 512 token. Bu denemede Türkçe belge 0,7953 ile birinci oldu. Üç skor da 0,74–0,80 arasında. Vektörleştirme 0,0560 saniye. |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 118M (471 MB) | Apache 2.0 | En hafif model. 384 boyut, en fazla 128 token. Hava cümlesini 0,0300'e indirdi. İngilizce belge 0,4654 ile birinci. |
| BAAI/bge-m3 | 568M (2,27 GB) | MIT | 1024 boyut, 8192 token. İngilizce belge 0,6639, hava cümlesi 0,4044. Aynı CPU'da daha ağır. |
E5, sorunun başına `query: `, belgenin başına `passage: ` yazılmasını istiyor. `embedding_test.py` bu önekleri ekliyor. MiniLM ve `bge-m3` önek istemiyor; `karsilastirma.py` onlara düz metni veriyor.

## Veri
| Rol | Metin |
| --- | --- |
| Soru | Sistemlerin yanıt süresi nasıl hızlandırılır? |
| Belge 1 | Yazılım mimarisinde mikroservisler performansı artırabilir. |
| Belge 2 | Reducing latency in systems requires query optimization and caching. |
| Belge 3 | Bugün hava güneşli ve sıcak olacak. |
Belge 1 aynı dilde ve konuyla ilişkilidir. Belge 2 başka dilde, doğrudan gecikme ve önbelleği anlatır. Belge 3 Türkçedir ve konu dışıdır.
## Yöntem
`SentenceTransformer` metni 768 boyutlu bir vektöre çevirir. `util.cos_sim` soru vektörü ile her belge vektörünün kosinüsünü hesaplar. Aynı yön 1'e yaklaşır. Betiğin yazdığı süre yalnızca bu dört metnin vektörleştirilmesidir. Modelin indirilmesi ve belleğe yüklenmesi bu sürenin dışındadır.
## Sonuç
Aşağıdaki sayılar, model belleğe alındıktan sonra CPU üzerinde alınan tek çalıştırmadır. Ortam: Python 3.12, `sentence-transformers` 6.3.0, `torch` 2.15.0+cpu. GPU yoktur.
| Belge | Skor |
| --- | --- |
| Belge 1, mikroservisler | 0,7953 |
| Belge 2, gecikme ve önbellek | 0,7627 |
| Belge 3, hava durumu | 0,7387 |
Vektörleştirme 0,0631 saniye sürdü.
Sıra beklenen yöndedir. İki performans cümlesi hava cümlesinin üstündedir. İngilizce gecikme cümlesi Türkçe soruya yakındır. Aralık dardır. E5, eğitimindeki düşük sıcaklık nedeniyle kosinüs skorlarını yaklaşık 0,7 ile 1,0 arasına yığar. Anlamlı olan mutlak değer değil, belgelerin birbirine göre sırasıdır.
## Neden bu model?
Seçim üç ölçüte göre yapıldı: Türkçe ve İngilizce desteği, bellek yükü, ticari kullanıma uygun açık lisans. Üç model de bu lisansa uyar. Ölçüm yalnızca seçilen modelle yapıldı; diğer ikisi bu depoda çalıştırılmadı.
Seçim üç ölçüte göre yapıldı: Türkçe ve İngilizce desteği, bellek yükü, ticari kullanıma uygun açık lisans. Üç model de bu lisansa uyar. Skorlar yalnızca seçilen modelden gelir. Diğer iki model bu depoda çalıştırılmadı.
Boyutlar Hugging Face üzerindeki ağırlık dosyasından okundu. `model.safetensors` veya `pytorch_model.bin` bayt cinsinden bölünüp parametre sayısına çevrildi.
Ağırlık, Hugging Face dosyasının bayt uzunluğudur. Parametre sayısı modelin kendi bildirdiği büyüklüktür.
| Model | Parametre | Ağırlık | Lisans | Karar |
| --- | --- | --- | --- | --- |
| **intfloat/multilingual-e5-base** | 278M | 1,11 GB | MIT | **Seçilen model.** 100 dil, 768 boyut, en fazla 512 token. `bge-m3` modelinden hafif, MiniLM'den uzun bağlamlı. Soru için `query: `, belge için `passage: ` öneki gerekir. |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 118M | 471 MB | Apache 2.0 | Daha az bellek kullanır. Vektör boyutu 384, bağlam en fazla 128 tokendir. Kısa cümlede ucuzdur; uzun metinde E5'in gördüğü kısmı keser. |
| BAAI/bge-m3 | 568M | 2,27 GB | MIT | 1024 boyut, 8192 token, yoğun, seyrek ve çoklu vektör aramasını birlikte yapar. Aynı CPU'da bellek ve süre maliyeti yüksektir. Önek istemez. |
| BAAI/bge-m3 | 568M | 2,27 GB | MIT | 1024 boyut, 8192 token; yoğun, seyrek ve çoklu vektör aramasını birlikte yapar. Aynı CPU'da bellek ve süre maliyeti yüksektir. Önek istemez. |
`multilingual-e5-base` 110M veya yaklaşık 450 MB değildir. Bu rakamlar MiniLM'e yakındır. E5'in taban sürümü `xlm-roberta-base` ağırlığındadır: 278.050.322 parametre, 1.112.201.288 bayt.
## Kurulum
GPU olmadığı için CPU tekerleği kullanılır.

[12 lines collapsed]

```
İlk çalıştırmada model Hugging Face'ten indirilir ve önbelleğe alınır. Sonraki çalıştırmalar yalnızca vektörleştirmeyi ölçer.
## Sınır
Tek soru ve üç belge vardır. Bu bir arama testi seti değildir. MiniLM ve `bge-m3` aynı metinlerle ölçülmedi. Süre, model yüklendikten sonraki vektörleştirmeyi gösterir.
