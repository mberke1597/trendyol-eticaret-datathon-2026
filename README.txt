Overview
2010 yılında kurulan Trendyol, Türkiye'nin lider, dünyanın önde gelen e-ticaret platformlarından biridir. Platform Türkiye, Almanya, Azerbaycan, Körfez Ülkeleri ve son olarak Romanya, Çekya, Slovakya, Polonya, Macaristan, Bulgaristan ve Yunanistan'da yerel dilde özel uygulamalar aracılığıyla 250 bin satıcı ve markayı 42 milyon müşteriyle buluşturuyor.

"Verilerle yaşıyoruz" mottosu Trendyol'un temel değerleri arasında yer alıyor. Bu motto sadece veri bilimi departmanında değil, şirketin tüm departmanlarda (İK, CRM vb.) uygulanıyor. Trendyol'un veri bilimi ekibi, çalışmalarında alışveriş deneyiminin her yönüne dokunan bir yaklaşımı benimsiyor. Bunların bir kısmını aşağıdaki listede görebilirsiniz:

Müşterinin ana sayfamızı açtığında gördüğü butikler ve bunların sıralaması
Müşterinin önceki davranışlarına bağlı olarak her müşteriye özel sunulan öneriler
Müşterinin görüntülediği ürünlere benzer öneriler
Müşterinin aradığı en alakalı ürünlerin gösterimi
Dolandırıcılık tespiti
Sohbet robotu
Talep tahmini
Satıcı puanlarının oluşturulması ve aynı ürünü satan farklı satıcıların sıralanması
Kargo gönderimi
Tüm bu süreçlerde regresyon, sınıflandırma, kümeleme, NLP ve görüntü işleme gibi makine öğrenimine ve optimizasyona dayalı teknolojileri her yönüyle kullanıyoruz. En güzel tarafı da geniş bir veri setimizin olması. Bu yıl düzenlediğimiz Datathon ile siz de bu zorlu, eğlenceli ve benzersiz deneyimin parçası olabilirsiniz.

Start

6 days ago
Close

15 days to go
Problem
Bu aşamada katılımcılar, Trendyol'un arama verilerini kullanarak bir arama teriminin bir ürünle ne kadar alakalı olduğunu belirlemek üzere çalışacaklar.

Amaç: Verilen bir (arama terimi, ürün) çifti için ürünün, arama terimiyle:

Alakalı (1 — relevant)
Alakasız (0 — irrelevant)
olup olmadığını tahmin eden bir binary model geliştirmektir. Katılımcılardan; arama terimi, ürün ismi, kategori hiyerarşisi ve detaylı ürün özelliklerini (materyal, renk, stil vb.) birlikte kullanan modeller geliştirmeleri beklenmektedir.

Eğitim verisi yalnızca pozitif çiftlerden oluşmaktadır. Test seti ise hem alakalı hem alakasız çiftleri içerir. Katılımcıların negatif durumları modele nasıl öğretecekleri tamamen kendi inisiyatiflerindedir.

Değerlendirme
Çözümlerin değerlendirilmesinde macro-averaged F1 metriği kullanılacaktır. İki sınıf (0 = irrelevant, 1 = relevant) için F1 skorları ayrı ayrı hesaplanıp eşit ağırlıklı ortalaması alınacaktır:


Her sınıf için:


Leaderboard, submissionlarınızdaki satırların Public subset'i üzerinde canlı olarak hesaplanır. Nihai sıralama yarışma sonunda Private subset'indeki örnekler üzerinden hesaplanan skora göre belirlenecektir.

Tahmin Formatı
Test setindeki (submission_pairs.csv) her satır bir (arama terimi, ürün) çiftini temsil eder ve bir id ile tanımlanır. Her id için ürünün arama terimiyle alakalı olup olmadığını 0 (irrelevant) veya 1 (relevant) olarak tahmin etmelisiniz. Submission dosyası sütun isimlerini içermeli ve sample_submission.csv ile birebir aynı formatta olmalıdır:

id,prediction
TST_a8a83d59d73cce,1
TST_5299616224ebbf,0
TST_465f5410a8a83d,1
prediction sütunu yalnızca 0 veya 1 değerlerini içermelidir.
Submission dosyası submission_pairs.csv ile aynı satır sayısına ve aynı id kümesine sahip olmalıdır.
Nihai Değerlendirme ve Kaggle Skorunun Ağırlığı
Yarışmanın nihai kazananları, ön değerlendirme ve hackathon sürecinin birleşik puanına göre belirlenir. Kaggle aşamasındaki private leaderboard skoru, nihai puanın %40'ını oluşturur. Ağırlıklandırma tablosu aşağıdaki gibidir:

Etki Oranı	Değerlendirme Kriteri
%40	Ön Değerlendirme Süreci: Kaggle Skoru
%20	Hackathon Süreci: Final Set Skoru
%10	Hackathon Süreci: Sunum Kalitesi
%10	Hackathon Süreci: Model Hızı
%10	Hackathon Süreci: Model Açıklanabilirlik Arayüzü
%10	Hackathon Süreci: Final Raporu
Kaggle Skorunun Hackathon Aşamasına Taşınması
Yalnızca Kaggle private leaderboard'da dereceye giren ilk 10 takımın skorları ikinci aşamaya taşınır. Kaggle başarısının genel skora etkisinin hem sıralama hem de ham skor tarafında eşit olması amacıyla bu 10 takımın skorlarına hibrit bir normalizasyon uygulanır:

Max normalizasyon: Her takımın skoru, birinci takımın (en yüksek) skoruna bölünür → MaxNorm = skor / en_yüksek_skor.
Sıralama çarpanı: Takımın 10'lu finalist sıralamasındaki yerine göre doğrusal olarak azalan bir çarpan uygulanır (1. takım = 1.0, 10. takım = 0.8). Bu, ham skorları çok yakın olan takımlar arasında sıralamanın da önemini tekrar ortaya çıkarır.
%40'a ölçekleme: Nihai Kaggle katkısı MaxNorm × Sıralama Çarpanı × 40 olarak hesaplanır (toplam puanın %40'lık dilimi olduğu için).
Yarışmaya 100 takımdan az aktif katılım (en az 1 submission yapan takım) olması durumunda Top 10 takım kendi içlerinde değil, 10 takımın da geçmeyi başardığı en iyi benchmark çözümün skoruna göre normalize edilir (sıralama çarpanının alt eşiği belirtilen benchmark skorunun sıralamasına denk gelmiş olur).

Aşağıdaki tablo, 10 finalist takım için normalizasyon mekaniğini örnek değerlerle göstermektedir:

Sıra	Private LB Skoru	Max Normalizasyon	Sıralama Çarpanı	Genel Skora Etki (MaxNorm × Sıralama Çarpanı × 40)
1	0.85321	1.00000	1.00000	40.00000
2	0.83904	0.98339	0.97778	38.46156
3	0.82451	0.96636	0.95556	36.93659
4	0.81002	0.94938	0.93333	35.44339
5	0.80337	0.94159	0.91111	34.31569
6	0.79588	0.93281	0.88889	33.16661
7	0.78112	0.91551	0.86667	31.73779
8	0.76540	0.89708	0.84444	30.30121
9	0.75470	0.88454	0.82222	29.09147
10	0.74125	0.86878	0.80000	27.80096




Dataset Description
Önemli Notlar
Tüm ürün (item_id) ve arama terimleri (term_id) anonimleştirilmiş formattadır. Hashler içerik veya sıralama hakkında hiçbir bilgi taşımaz.
training_pairs.csv yalnızca pozitif (relevant, label = 1) çiftler içerir. Tahmin edilmesi gereken test verisinde negatif çiftler de bulunmaktadır. Yarışmacılar eğitimde negatif örnek kullanmak istedikleri takdirde kendileri üretmelidir.
gender, age_group gibi bazı kolonlarda unknown veya boş değerler bulunabilir.
Dosyalar
items.csv
Açıklama: Ürün kataloğu.

Kolon Adı	Veri Tipi	Açıklama
item_id	string	Hash tabanlı ürün kimliği (örn. ITEM_7d18e2e51ef1)
title	string	Ürün başlığı
category	string	Ürün kategori zinciri (örn. ayakkabı/spor ayakkabı/sneaker)
brand	string	Ürünün markası
gender	string	Ürüne tanımlı cinsiyet etiketi (kadın, erkek, unisex, unknown)
age_group	string	Ürüne tanımlı yaş grubu (yetişkin, çocuk, bebek, unknown)
attributes	string	anahtar: değer, ... formatında ürün özellikleri.
terms.csv
Açıklama: Arama terimi kataloğu.

Kolon Adı	Veri Tipi	Açıklama
term_id	string	Hash tabanlı arama terimi kimliği (örn. TERM_ccfefd8a)
query	string	Arama terim metni
training_pairs.csv
Açıklama: Eğitim için kullanılabilecek hazır pozitif terim-ürün çiftleri.

Kolon Adı	Veri Tipi	Açıklama
id	string	Çift kimliği (örn. TRN_23f5d7fbae)
term_id	string	terms.csv'ye referans
item_id	string	items.csv'ye referans
label	int64	Her zaman 1 (relevant)
submission_pairs.csv
Açıklama: Tahmin yapılacak test terim-ürün çiftleri.

Kolon Adı	Veri Tipi	Açıklama
id	string	Çift kimliği (örn. TST_a8a83d59d73cce) — submission'da bu hashler kullanılmalıdır
term_id	string	terms.csv'ye referans
item_id	string	items.csv'ye referans
sample_submission.csv
Açıklama: Örnek submission dosyası (temsilen tüm tahminler 0).

Kolon Adı	Veri Tipi	Açıklama
id	string	submission_pairs.csv'deki her çift için hash
prediction	int64	Tahmin (Örnek dosyada tümü 0)

Data Dir path : "C:\Users\berke\Desktop\TrendyolE-Ticaret\trendyol-e-ticaret-yarismasi-2026-kaggle"


Buradan Başlayabilirsiniz!
Selamlar! 3 haftalık rekabetçi, eğitici ve eğlenceli geçmesini umduğumuz başka bir serüvenimize hoşgeldiniz. 😄

İlk olarak "Team" sekmesine giderek yarışmaya katılmış diğer takım arkadaşlarınızı davet ederek başvururken taahhüt ettiğiniz takımınızı oluşturmanız gerekmektedir. Lütfen bu aşamayı hallettiğinize emin olun.

Yarışmamız ile ilgili detayları, problemi ve değerlendirme kriterini "Overview" sekmesi altından inceleyebilirsiniz.

Yarışmamızda size sağladığımız bütün verilere ait açıklamaları "Data" sekmesi altından ulaşabilirsiniz.

Yarışmaya daha hızlı alışmanıza ve çözümler oluşturmanıza yardımcı olacak kod örneklerini paylaştık. Maalesef Kaggle private competition kuralları gereği Code sekmesi üzerinden başkasının görebileceği şekilde kaynak paylaşımı hala mümkün değil gibi. Sizler için hazırladığımız örnek baseline çözüme buradan ulaşabilirsiniz:

Bu notebook, yarışma verisinin kullanımı ve basit bir word co-occurrence mantığı ile tahmin yapılmasını içermektedir.

Notebooktaki örnek kod; sorgu ile ürün en az bir ortak kelimeyi paylaşıyorsa alakalı (1), paylaşmıyorsa alakasız (0) olarak etiketleme yaklaşımını simüle etmektedir.

from pathlib import Path

DATA_DIR = Path("/kaggle/input/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle")

MIN_TOKEN_LEN = 2   # bu uzunluktan kısa kelimeleri yok say (tek karakterleri eler)
MIN_OVERLAP   = 1   # 'relevant' demek için gereken ortak kelime sayısı

assert (DATA_DIR / "submission_pairs.csv").exists(), f"veri bulunamadı: {DATA_DIR.resolve()}"
import pandas as pd

pairs = pd.read_csv(DATA_DIR / "submission_pairs.csv")   # id, term_id, item_id
terms = pd.read_csv(DATA_DIR / "terms.csv")             # term_id, query
items = pd.read_csv(DATA_DIR / "items.csv")             # item_id, title, category, ...

test = pairs.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
print(f"{len(test):,} çift yüklendi")
test.sample(5, random_state=42)
3,359,679 çift yüklendi
id	term_id	item_id	query	title	category	brand	gender	age_group	attributes
1160283	TST_104765a215fac0	TERM_1d855961	ITEM_7c79134be0e7	ahşap top kapak	dekoratif kırmızı elma set	ev & mobilya/ev/ev dekorasyon/dekoratif obje v...	hesabınca	unknown	unknown	materyal: polyester, renk: kırmızı, boyut/ebat...
2253359	TST_1d98bfc10f53f9	TERM_7d8e6066	ITEM_c59aa11af10b	erkek çocuk outdoor bot	weather forecast wf çelik uçlu mavi kedi köpek...	süpermarket/pet shop/kedi ürünleri/kedi makası	life petmarket	unknown	unknown	menşei: tr, bakım talimatları (genel): ürünün ...
2799740	TST_9dc3efbda809ab	TERM_4573472a	ITEM_52c14f939194	çiğköfte baharatı	leila kiremit renk yatak odası, mutfak, yemek ...	ev & mobilya/mobilya/elektrik & aydınlatma/avize	decory	unknown	unknown	materyal: plastik, model: modern, duy tipi: e2...
568555	TST_ae702c8e9ce31f	TERM_b9efbf1d	ITEM_30c55ec44b37	shea butter	şampuan ve saç bakım kremi kepeğe karşı etkili...	kozmetik & kişisel bakım/saç bakım/şampuan	elidor	kadın	yetişkin	özellik: sülfatsız, etki: kepek önleyici, haci...
2493674	TST_47d25c75035b23	TERM_0b12f7c6	ITEM_71f3fc04574f	erkek çocuk paten	kartela kanatlı at kız çocuk erkek çocuk oyunc...	anne & bebek & çocuk/oyuncak/figür oyuncaklar/...	paraply	unisex	bebek & çocuk	paket içeriği: 1'li, renk: karışık, yaş: 1+ ya...
import re
import numpy as np

_TOKEN_SPLIT_RE = re.compile(r"[^0-9a-zçğıöşü]+")

def _norm(s):
    if not isinstance(s, str):
        return ""
    s = s.replace("İ", "i").replace("I", "ı")
    return s.lower().replace("i̇", "i")

def tokenize(s, min_len=MIN_TOKEN_LEN):
    return {t for t in _TOKEN_SPLIT_RE.split(_norm(s)) if len(t) >= min_len}

queries = test["query"].tolist()

# Ürün metni: başlık + kategori + marka + cinsiyet + yaş grubu + özellikler
item_texts = (
    test["title"].fillna("") + " " + test["category"].fillna("") + " "
    + test["brand"].fillna("") + " " + test["gender"].fillna("") + " "
    + test["age_group"].fillna("") + " " + test["attributes"].fillna("")
).tolist()

n = len(queries)
overlap = np.empty(n, dtype=np.int32)
for i in range(n):
    q = tokenize(queries[i])
    overlap[i] = len(q & tokenize(item_texts[i])) if q else 0

# Ortak kelime sayısı eşiği geçiyorsa 'relevant' olarak işaretlenir
pred = (overlap >= MIN_OVERLAP).astype(np.int8)
submission = pd.DataFrame({"id": test["id"].values, "prediction": pred})
submission.to_csv("submission.csv", index=False)
