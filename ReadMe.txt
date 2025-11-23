## Açıklama 
Matplotlib, numpy ve OpenCV kullanılarak
geliştirilen hareket takip modeli. Kod modüler yapıda organize edilmiştir.


## Gereksinimler 
pip install -r requirements.txt 
|-"-videolar" |-"videolar" |-"test{file_index}.mp4"

## İsteğe bağlı
|-"referans{file_index}.txt"

Bu dosya kıyaslamak için kullanılacak dosyadır. Kullanılması durumunda ground truthlar ekrana yazdırılır

## Çalıştırma 
 python HW2.py

### Dosya Düzeni 
24501104_odev2/
|- "python HW2.py"
|- "requirements.txt"
|- "odev2-videolar" |-"odev2-videolar" |-"test{file_index}.mp4"
|- "ReadMe.txt"

## Süreç

Önce düşüş anı tespiti yapılır. Düşüş anı tespitinden sonra hareket analizi ekranı'da ekranda belirir.

Hareket analizi, kırmızı/ yeşil cell renkleri ve mavi rover bounding boxlarını içeren bir arayüzdür.

Referans dosyasının dahil edilmesiyle görselleştirme güçlendirilir.

Hareket analizinden sonra debug amacıyla optik flow büyüklükleri ve koda girilmiş olan (1.) cell'in ground truthları görselleştirilir.