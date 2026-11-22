# radionc-ct

Tedavi öncesi planlama BT'sinden radyomik özellik çıkarımı. Hasta klasörleri çalışma kimliği taşır (`P001`, `LUNG1-001`); ad, protokol numarası veya doğum tarihi klasör adına yazılmaz. Hasta görüntüleri ve türetilmiş tablolar git'e girmez.

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

## Önceki script'e göre değişenler

- **Kontur doldurma düzeltildi.** Eski yöntem (PIL poligon) sınır piksellerini de dolduruyordu; bu, hacmi yaklaşık %9 şişiriyor ve şekil özelliklerini bozuyordu. Artık piksel merkezi poligonun içindeyse maskeye girer (fantom testinde teorik hacimden %2'den az sapar).
- Tek hasta yerine klasördeki bütün hastalar; sabit `P001` kaldırıldı. Kesilen koşu kaldığı yerden devam eder.
- Yalnızca shape/firstorder/GLCM değil, rapordaki yedi sınıf.
- Ayarlar koddan YAML'a taşındı.
- Seri/RTSTRUCT seçimi dosya adına değil başlık bilgisine göre yapılır.

`data/metadata/P001_radiomics.csv` eski yöntemle üretilmişti; yeni çıktıyla değerleri farklı olacaktır.

## Testler

`pytest`: HU dönüşümü, piksel↔mm dönüşümü, fantom küre hacmi, iç içe poligon, halka kalınlığı, kimlik denetimi, ROI adı eşleme, hasta keşfi ve tablo birleştirme. PyRadiomics fantom testleri (küre hacmi, sabit bölgede entropi 0) `pyradiomics` ve `SimpleITK` kuruluysa çalışır, değilse atlanır. DICOM okuma gerçek veriyle denenmelidir.
