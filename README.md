# El İzi ile Sistem Açma

Sol elinizi okutarak video oynatımını tetikleyen basit bir el tanıma sistemi.

## Özellikler

- 🎯 **Sol El Tanıma**: Sadece sol el kabul edilir
- 🖼️ **El İskeleti Görselleştirmesi**: Kırmızı noktalar ve beyaz çizgiler
- 🟢 **Doğrulama İlerlemesi**: Yeşil dairesel ilerleme göstergesi
- 🎬 **Otomatik Video Oynatma**: Doğrulama başarılı olunca video açılır

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python main.py
```

## Kullanım

1. Uygulamayı başlatın
2. Sol elinizi şablon üzerine yerleştirin
3. 2 saniye boyunca konumunu koruyun
4. Doğrulama tamamlandığında video otomatik oynatılır
5. Çıkmak için **ESC** tuşuna basın

## Dosya Yapısı

```
├── main.py                  # Ana program
├── hand_template.png        # El şablon görseli
├── hand_landmarker.task     # MediaPipe model
├── requirements.txt         # Python bağımlılıkları
├── video/
│   └── login.mp4           # Oynatılacak video
└── README.md
```

## Sistem Gereksinimleri

- Python 3.8+
- Kamera (Webcam)
- macOS / Linux / Windows
