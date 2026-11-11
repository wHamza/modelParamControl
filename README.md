# modelParamControl

CIFAR-10 üzerinde küçük bir CNN eğitimi etrafında kurulmuş bir deney takibi ve tekrar üretilebilirlik çalışması. Amaç yüksek doğruluk değil; her sonucun hangi koddan, hangi ayarlardan ve hangi seed'den geldiğinin izlenebilir olması.

## Özet

- Her çalıştırma config'i, seed'i, git commit hash'ini, torch sürümünü ve GPU adını W&B'ye kaydeder.
- Aynı commit, config ve seed ile tekrarlanan çalıştırma birebir aynı sonucu verdi.
- 5 seed'lik deneyde test doğruluğu 0.745 ile 0.752 arasında kaldı.
- 20 çalıştırmalı rastgele aramada learning rate belirleyici, weight decay etkisiz çıktı. Hiçbir config baseline'ı gürültü payının ötesinde geçemedi.

## Kurulum

| Bileşen | Değer |
|---|---|
| Veri | CIFAR-10 (Hugging Face `uoft-cs/cifar10`), augmentation yok |
| Model | `SmallCNN`: 3 blok (Conv, BatchNorm, ReLU, MaxPool), global average pooling, Linear; `width=32` ile 94.762 parametre |
| Optimizer | SGD, momentum 0.9 |
| Schedule | `OneCycleLR`, batch başına adım |
| Baseline config | lr 0.1, batch size 128, 5 epoch, weight decay 5e-4, seed 42 |
| Donanım | Google Colab GPU |

## Çalıştırma

```bash
pip install -r requirements.txt
wandb login

python train.py                                   # baseline
python train.py --seed 0 --group baseline-seeds   # seed deneyi
wandb sweep sweep.yaml                            # sweep oluştur
wandb agent <kullanıcı>/exp-tracking/<sweep-id> --count 20
```

Colab'de `run.ipynb` repoyu klonlar, bağımlılıkları kurar ve `train.py`'yi çalıştırır. Eğitim kodu notebook'ta değil `train.py`'de durur; böylece her sonuç bir commit'e karşılık gelir.

## Neler loglanıyor

- **Config:** `config.yml`'deki tüm hiperparametreler.
- **Ortam:** git commit hash, torch sürümü, cihaz, GPU adı.
- **Epoch metrikleri:** train loss, test loss, test accuracy, learning rate, ortalama gradient norm.
- **Örnek tahminler:** eğitim sonunda 16 test görüntüsü, tahmin ve gerçek sınıf adlarıyla.

Commit edilmemiş değişiklik varken çalıştırılırsa hash'in sonuna `-dirty` eklenir. `-dirty` taşıyan bir sonuç yeniden üretilebilir sayılmaz.

## Tekrar üretilebilirlik

Sabitlenen rastgelelik kaynakları: Python `random`, NumPy, PyTorch ve DataLoader'ın karıştırma sırası (ayrı bir `torch.Generator` ile).

Test: aynı commit, config ve seed ile iki ayrı çalıştırma yapıldı ve sonuçlar birebir aynı çıktı.

## Seed deneyi
rastgele 5 seed ile deney yapildi

- Aralık: 0.745 - 0.752
- Ortalama ± standart sapma (`ddof=1`): TODO

Bu sapma, tek çalıştırmalık karşılaştırmaların gürültü tabanıdır. İki config'in tek çalıştırmaları karşılaştırılırken farkın sapması bunun yaklaşık 1.4 katıdır; kabaca 0.8 puanın altındaki farklar seed şansıyla açıklanabilir.

## Hyperparameter sweep

Tuning Playbook'a göre mimari ve batch size sabit tutuldu; learning rate ve weight decay logaritmik ölçekte rastgele arandı. Toplam 20 çalıştırma.

| Parametre | Dağılım | Aralık |
|---|---|---|
| lr | log-uniform |  (`sweep.yaml`'daki min - max) |
| weight_decay | log-uniform |  (`sweep.yaml`'daki min - max) |

### En iyi 3 config

| Sıra | lr | weight decay | Test accuracy |
|---|---|---|---|
| 1 | 0.0877 | 0.00003446 | 0.7489 |
| 2 | 0.06717 | 0.00006276 | 0.7429 |
| 3 | 0.05489 | 0.00001582 | 0.7363 |

### Bulgular

- **Learning rate belirleyici.** En iyi çalıştırmaların hepsi lr aralığının üst kısmında, en kötüleri alt kısmında.
- **Weight decay bu kurulumda fark etmiyor.** İyi sonuçlar weight decay ekseninin her yerine dağılmış. 5 epoch'luk kısa eğitimde beklenen bir sonuç.
- **Fark gürültüden büyük mü?** 1. ve 3. config arasındaki 1.3 puanlık fark gürültü eşiğinin üstünde; büyük ihtimalle gerçek. 1. ve 2. arasındaki 0.6 puanlık fark eşiğin altında; hangisinin daha iyi olduğu tek çalıştırmayla söylenemez.
- **Sweep baseline'ı geçemedi.** En iyi sonuç (0.7489) baseline'ın seed aralığının (0.745 - 0.752) içinde.

### Sınırlamalar

- En iyi sonuçlar lr aralığının üst sınırında çıktı ve baseline'ın lr değeri (0.1) aralığın dışında kaldı. Tepe noktası aranan bölgenin ötesinde olabilir; sonraki adım lr için daha yüksek bir aralıkta ikinci bir sweep.
- En iyi 3 config tek seed ile çalıştırıldı. Aralarındaki sıralamayı kesinleştirmek için her birinin birkaç seed ile tekrarlanması gerekir.


## Bağlantılar

- W&B projesi: https://forge.coreweave.com/wandb/whamza-dmr-necmettin-erbakan-niversitesi/exp-tracking?nw=nwuserwhamzadmr