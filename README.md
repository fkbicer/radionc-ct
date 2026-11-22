# radionc-ct

Tedavi öncesi planlama BT'sinden radyomik özellik çıkarımı ve klinik değişkenlerle birleştirme. Hasta klasörleri çalışma kimliği taşır (`P001`, `LUNG1-001`); ad, protokol numarası veya doğum tarihi klasör adına yazılmaz. Hasta görüntüleri ve türetilmiş tablolar git'e girmez.

## Nereye koyulur

Bir hastanın elindeki bütün `.dcm` dosyaları, ayrıştırmadan:

`data/raw/P001/inbox/`

Bu klasör kasıtlı olarak karışıktır: CT kesitleri ve kontur dosyası aynı uzantıyı kullanır, dosya adı bir şey söylemez. Alt klasör olabilir.

## Klasörler

```text
data/
  raw/<id>/inbox/          gelen DICOM, olduğu gibi
  features/                extract çıktısı (git dışı)
    per_patient/           hasta başına özellik satırı ve denetim kaydı
    radiomics_wide.csv     hasta başına bir satır
    audit.csv              her hastanın durumu ve sorunları
  sorted/, masks/          iskelet (git'te .gitkeep olarak durur)
  metadata/patients.csv    kimlik içermeyen hasta listesi
```

## Çalıştırma

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "numpy>=1.23,<2" wheel
pip install -r requirements.txt --no-build-isolation   # pyradiomics derlemesi için
pip install -e .

python -m rad extract --patients P001     # önce tek hasta
python -m rad extract --workers 4         # data/raw altındaki hepsi, kesilirse devam eder
python -m rad combine                     # özet tabloları yeniden üretir
python -m rad dataset                     # klinik + etiket + radyomik -> data/features/dataset.csv
python -m rad select                      # seçim zincirinin tüm veride neyi tuttuğunu gösterir (yalnızca inceleme)
python -m rad run                         # model merdiveni x modeller, tekrarlı iç içe CV -> outputs/<zaman>/ (+ report.html)
python -m rad explain --set M6 --model logreg   # katsayılar / SHAP
python -m rad report                      # son koşunun report.html dosyasını yeniden üretir
pytest
```

`extract` bittiğinde önce `data/features/audit.csv` dosyasına bakın: sorunlu hastalar özellik matrisine girmez ve gerekçesiyle burada listelenir.

## Ne yapar (rapor 4.2–4.4)

- Planlama BT'si, inbox'taki en büyük CT serisi olarak seçilir. RTSTRUCT, aynı referans çerçevesine bakan dosyadır.
- **Kimlik denetimi:** PatientID izin verilen desene (`dicom.patient_id_pattern`) uymazsa ya da doğum tarihi/adres gibi alanlar doluysa işlem durur, hasta `audit.csv`'ye yazılır.
- **HU:** Kesit başına eğim/kesişimle dönüştürülür. Okunamayan kesitler komşularından doldurulur (`dicom.missing_slice`) ve sayısı `audit.csv`'de görünür. Önceki script bu kesitleri hava olarak bırakıyordu.
- **Konturlar:** ROI adları normalize edilip birebir eşlenir (`Lung-GTV` ≠ `GTV`). Periferi konturu yoksa GTV'den 5 mm halka türetilir; halka GTV'yi içermez. İç içe poligonlar delik olarak işlenir.
- **Maske denetimi:** boş maske, hacim aralığı, kesit ızgarası dışında kalan poligonlar, GTV–periferi çakışması.
- **PyRadiomics:** shape, firstorder, GLCM, GLRLM, GLSZM, NGTDM, GLDM; 2 mm izotropik B-spline yeniden örnekleme, `binWidth=25`. Hepsi `configs/default.yaml` içinde (rapor Ek B). Sütunlar `<roi>__<özellik>`.

## Klinik veri ve etiket (rapor 4.5, FR7)

`lung1_stage3_cohort.csv` (yolu `paths.clinical_csv`) hasta başına bir satır; `python -m rad dataset` onu radyomikle birleştirip `data/features/dataset.csv` yazar (git dışı) ve sayıları yazdırır. Radyomik henüz çıkarılmadıysa yalnızca klinik sütunlar yazılır.

- **Özellikler:** yaş, T, N, cinsiyet, histoloji; sütunlar `clin__*`, radyomik sütunlar `<roi>__*`. Eksik değerler bu aşamada **doldurulmaz**; eğitim katında doldurulacaktır.
- **Etiket:** `label_2yr` (1 = 730 günden önce öldü, 0 = 730 günden uzun izlendi). Etiket, `Survival.time` ve `deadstatus.event` ile yeniden hesaplanıp karşılaştırılır; çelişki varsa koşu durur. 730 günden önce sansürlenen hasta etiketlenemez ve dışarıda kalır (Lung1 stage III'te böyle hasta yok).
- **Sızıntı koruması:** `Survival.time`, `deadstatus.event` ve etiket sütunu özellik olarak verilirse `LeakageError` fırlatılır.
- **Kullanılmayanlar:** `Clinical.M.Stage` (287 hastanın 282'sinde sıfır) ve `Overall.Stage` (T ve N'den türetilmiş, bilgi eklemez).
- **Birleştirme:** hasta kimliği ile iç birleşim. Etiketi olup radyomiği olmayan ve radyomiği olup etiketi olmayan hasta sayıları yazdırılır (kimlikler değil, yalnızca sayılar).

## Özellik seçimi (rapor 4.6, `src/rad/selection.py`)

Radyomik sütunlar sırayla: eksik oranı filtresi → medyan doldurma → neredeyse sabit filtresi → (ICC kararlılık filtresi, ICC tablosu verilirse) → z-skoru → korelasyon kümeleri (|ρ|>0.9; her kümeden etiketle en çok ilişkili olan kalır) → elastic-net ile en çok 15 özellik. Klinik sütunlar (`clin__*`) yalnızca doldurulur ve ölçeklenir.

Hepsi scikit-learn dönüştürücüsüdür ve yalnızca `fit` sırasında öğrenir; bir eğitim katında `fit` edildiğinde test katı hiçbir aşamayı etkilemez. `python -m rad select` zinciri tüm veride çalıştırıp seçilenleri yazdırır: bu yalnızca inceleme içindir, performans tahmini değildir. Model karşılaştırması sonraki PR'larda, seçim her katta yeniden yapılarak yapılacak.

## Model ve doğrulama (rapor 4.7–4.8) — lojistik regresyon, doğrusal SVM, random forest, XGBoost

`python -m rad run` her merdiven kümesi (M0 önsel, M1 klinik, M2 GTV, M5 GTV+periferi, M6 hepsi) için elastic-net lojistik regresyonu **tekrarlı, katmanlı, iç içe çapraz doğrulama** ile değerlendirir (5 kat × 10 tekrar, `configs/default.yaml`). Radyomik henüz çıkarılmadıysa yalnızca radyomik grubu gerektirmeyen kümeler (M0, M1) çalışır; eksik gruplu küme başka adla değil, atlanarak geçilir.

- Dış döngü yalnızca performansı ölçer. Her dış eğitim kümesinde iç döngü `C` ve `l1_ratio` seçer; doldurma, ölçekleme, özellik seçimi ve model o kümede yeniden öğrenilir. Bölme hasta düzeyindedir.
- Çıktılar (`outputs/<zaman>/`, git dışı): `metrics.csv` (AUC, dengeli doğruluk, makro-F1, Brier, ECE, duyarlılık, özgüllük; tekrarlar üzerinde ortalama ve sd), `oof_predictions.csv`, `selected_features.csv` (kat başına seçilme sıklığı), `run.json` (yapılandırma, seed, git commit, sürümler).
- **Modeller:** elastic-net lojistik regresyon (`C`, `l1_ratio` ızgarası) ve doğrusal SVM (`C` ızgarası; olasılıklar Platt kalibrasyonuyla, `CalibratedClassifierCV`). Random forest sığ ağaçlıdır (300 ağaç, `max_depth` ve `min_samples_leaf` ızgarası). XGBoost sığ ağaçlı ve güçlü düzenlidir (200 ağaç, `max_depth` 2–3, `learning_rate` ızgarası, `reg_lambda=5`, alt örnekleme 0.8). Lojistik regresyon, SVM ve random forest sınıf ağırlıklıdır.
- **Güven aralığı:** tekrarlar üzerinden ortalanan out-of-fold tahminlerde hasta düzeyinde bootstrap (1000 çekiliş, yüzdelik aralık); `metrics.csv`'de `auc_ci_low/high`. Yalnızca hasta örneklemesinden gelen belirsizliği ölçer; CV tekrarları arasındaki oynaklık `auc_sd` sütunundadır.
- **Permütasyon testi:** `--permutations N` etiketleri karıştırıp **bütün boru hattını** (seçim dahil) N kez yeniden koşar; `permutation_p` sütunu. Null dağılım `permutation_repeats` (varsayılan 1) CV tekrarıyla üretilir, gözlenen tahminden daha gürültülüdür, bu da testi tutucu yapar. Rapor 1000 permütasyon önerir; yavaştır, son koşuya bırakın.
- **Küme karşılaştırması:** `comparisons.csv`, aynı hastalar üzerinde eşli bootstrap ile AUC farkı, %95 aralığı ve p değeri; Holm düzeltmeli `p_holm`. Karşılaştırmalar: M1–M0 (klinik önselden iyi mi?), M2–M1 (BT radyomiği klinikten iyi mi?), M5–M2 (periferi katkı sağlıyor mu?), M5–M1, M6–M5 (klinik ek katkı sağlıyor mu?), M6–M1. İlgili kümeler çalışmadıysa (ör. radyomik yok) o karşılaştırma yapılmaz.
- Denenen **bütün** yapılandırmalar `metrics.csv`'ye yazılır (raporlama kuralı: yalnızca en iyisi değil).

## Önceki script'e göre değişenler

- **Kontur doldurma düzeltildi.** Eski yöntem (PIL poligon) sınır piksellerini de dolduruyordu; bu, hacmi yaklaşık %9 şişiriyor ve şekil özelliklerini bozuyordu. Artık piksel merkezi poligonun içindeyse maskeye girer (fantom testinde teorik hacimden %2'den az sapar).
- Tek hasta yerine klasördeki bütün hastalar; sabit `P001` kaldırıldı. Kesilen koşu kaldığı yerden devam eder.
- Yalnızca shape/firstorder/GLCM değil, rapordaki yedi sınıf.
- Ayarlar koddan YAML'a taşındı.
- Seri/RTSTRUCT seçimi dosya adına değil başlık bilgisine göre yapılır.

`data/metadata/P001_radiomics.csv` eski yöntemle üretilmişti; yeni çıktıyla değerleri farklı olacaktır.

## Testler

`pytest`: HU dönüşümü, piksel↔mm dönüşümü, fantom küre hacmi, iç içe poligon, halka kalınlığı, kimlik denetimi, ROI adı eşleme, hasta keşfi ve tablo birleştirme. Klinik tablo, etiket kuralı, sızıntı koruması, birleştirme ve gerçek Lung1 tablosunun kontrolü de test edilir. PyRadiomics fantom testleri (küre hacmi, sabit bölgede entropi 0) `pyradiomics` ve `SimpleITK` kuruluysa çalışır, değilse atlanır. DICOM okuma gerçek veriyle denenmelidir.
