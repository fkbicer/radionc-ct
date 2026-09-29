# radionc-ct

Oligometastaz riski için tedavi öncesi BT radyomiği. Hasta klasörleri çalışma kimliği taşır (`P001`); ad, protokol numarası veya doğum tarihi klasör adına yazılmaz.

## Nereye koyulur

Bir hastanın elindeki bütün `.dcm` dosyaları, ayrıştırmadan:

`data/raw/P001/inbox/`

Bu klasör kasıtlı olarak karışıktır. Dosya adı kontur olup olmadığını söylemez. Görüntüler ve konturlar aynı uzantıyı kullanır.

## Klasörler

```text
data/
  raw/P001/inbox/          gelen DICOM, olduğu gibi
  sorted/P001/
    ct/                    planlama BT kesitleri
    rtstruct/              kontur dosyası, varsa tek (veya birkaç) dosya
    rtdose/                doz hacmi, varsa
    rtplan/                plan, varsa
    other/                 bunlara girmeyen seriler
  masks/P001/
    gtv/                   primer tümör maskesi
    peritumoral_5mm/       GTV + 5 mm
    ctv_lenfatik/          mediastinal tutulu nod konturu
    lung_gtv/              aynı taraf akciğer eksi GTV
  metadata/patients.csv    kimlik içermeyen hasta listesi
```

`sorted/` ve `masks/` inbox okunmadan doldurulmaz. GTV, peritumoral alan, CTV lenfatik ve Lung-GTV ayrı DICOM serisi değildir; kontur setinin içindeki bölge adlarıdır. Peritumoral halka dosyada yoksa GTV’den 5 mm genişletilerek üretilir.

Ham ve türetilmiş görüntü git’e girmez. Takip edilenler klasör iskeleti ve `patients.csv` satırıdır.

## İlk çıkarım

`scripts/extract_radiomics.py` P001 planlama BT’sinden GTV ve peritumoral konturu okur. PyRadiomics ile şekil, yoğunluk ve GLCM doku özelliklerini `data/metadata/P001_radiomics.csv` dosyasına yazar.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install "numpy>=1.23,<2" wheel
pip install -r requirements.txt --no-build-isolation
python scripts/extract_radiomics.py
```
