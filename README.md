# Pekiştirmeli Öğrenme ile Satranç 🏁♟️

Bu proje, **Q-Öğrenme (Q-Learning)** algoritması kullanılarak geliştirilmiş bir satranç oyunudur. Oyuncu, beyaz taşlarla hamle yaparken, siyah taşlar bir basit yapay zeka tarafından kontrol edilir. Eğitim tamamlandıktan sonra, Pygame tabanlı bir GUI üzerinden oyunu oynayabilirsiniz.

---

## 📋 İçindekiler
- [Kurulum](#kurulum)
- [Nasıl Çalıştırılır?](#nasıl-çalıştırılır)
- [Yapı ve Teknolojiler](#yapı-ve-teknolojiler)
- [Özellikler](#özellikler)
- [Katkı](#katkı)
- [Lisans](#lisans)

---

## 🛠️ Kurulum

### Gereksinimler
- Python 3.7+
- Gerekli kütüphaneler:
  ```bash
  pip install pygame chess numpy
Resim Dosyaları
chess_pieces adında bir klasör oluşturun.

Aşağıdaki isimlerde satranç taşı resimlerini bu klasöre ekleyin:

Beyaz taşlar: white_pawn.png, white_rook.png, white_knight.png, white_bishop.png, white_queen.png, white_king.png

Siyah taşlar: black_pawn.png, black_rook.png, black_knight.png, black_bishop.png, black_queen.png, black_king.png

⚠️ Not: Resim dosyaları bulunamazsa, GUI taşları göstermeyecektir.

🚀 Nasıl Çalıştırılır?
Terminali açın ve proje dizinine gidin.

Aşağıdaki komutu çalıştırın:

bash
Copy
python rl_chess.py
Eğitim süreci başlayacaktır (5000 episode). Eğitim tamamlandığında GUI otomatik açılır.

GUI üzerinden:

Beyaz taşlarla hamle yapmak için sol tık kullanın.

Siyah taşlar otomatik hamle yapar.

## 🧠 Q-Learning Hakkında

### ❓ Q-Learning Nedir?
**Q-Learning**, pekiştirmeli öğrenme (reinforcement learning) algoritmalarından biridir. Bir ajanın, belirli bir durumda en uygun aksiyonu seçmeyi öğrenmesini sağlar. Temel mantık, her durum-aksiyon çifti için bir **Q-değeri** tutmak ve bu değerleri deneyimlerle güncellemektir. Q-Learning, **deneme-yanılma** ve **ödül maksimizasyonu** üzerine kuruludur.

---

### ⚙️ Projede Q-Learning Nasıl Uygulandı?
1. **Durum Temsili**:  
   Tahta pozisyonu, FEN (Forsyth-Edwards Notation) formatında temsil edilir. Örnek: `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR`.

2. **Aksiyon Uzayı**:  
   Her durumda geçerli hamleler (`legal_moves`) aksiyon olarak tanımlanır. Örneğin, `e2e4` (piyon ilerletme) veya `g1f3` (at hareketi).

3. **Ödül Fonksiyonu**:  
   - **Materyal Denge**: Taşların değerleri (piyon=1, vezir=9 vb.) üzerinden hesaplanır.  
   - **Pozisyonel Avantaj**: Taşların merkeze yakınlığına göre bonus puan.  
   - **Hamle Kalitesi**: Şah çekme, taş alma veya merkeze hamle gibi taktikler ek ödül getirir.  
   - **Oyun Sonu**: Mat durumunda ±100, beraberlikte 0 ödül.

4. **Temel Parametreler**:  
   - `epsilon`: Keşif (exploration) oranı. Başlangıçta yüksek (`1.0`), zamanla azalır (`0.01`).  
   - `alpha`: Öğrenme oranı (`0.1`).  
   - `gamma`: Gelecek ödüllerin indirim faktörü (`0.9`).  

---

### 📈 Q-Learning'in Projedeki Rolü
- **Beyaz Taşlar (İnsan/Kullanıcı)**: Q-Learning ajanı tarafından eğitilir.  
- **Siyah Taşlar (AI)**: Basit bir kural tabanlı algoritma (`SimpleBlackPlayer`) ile kontrol edilir.  
- **Eğitim**: 5000 episode boyunca Q-tablosu güncellenir. Her episode, bir satranç oyununu temsil eder.

---

### ⚠️ Sınırlamalar ve İyileştirme Alanları
- **Durum Uzayı Büyüklüğü**: Satrançta olası durum sayısı çok yüksek olduğundan, Q-tablosu verimsiz olabilir.  
  - Çözüm Önerisi: Derin Q-Networks (DQN) veya durum vektörleştirme.  
- **Ödül Dağılımı**: Seyrek ödül problemi (örn., mat durumuna kadar ödül alınamaması).  
  - Çözüm Önerisi: Ara ödülleri artırmak veya Monte Carlo Tree Search (MCTS) entegrasyonu.

---

### 📚 Kaynaklar
- [Q-Learning: A Beginner's Guide](https://www.geeksforgeeks.org/q-learning-in-python/)  
- [Satranç ve Pekiştirmeli Öğrenme](https://towardsdatascience.com/reinforcement-learning-for-chess-9658629f002)  

🧠 Yapı ve Teknolojiler
Ana Bileşenler
ChessEnvironment: Tahta durumunu, ödül hesaplamalarını ve oyun kurallarını yönetir.

QLearningAgent: Q-Öğrenme algoritmasını uygular.

SimpleBlackPlayer: Siyah taşlar için basit bir hamle seçici.

ChessGUI: Pygame tabanlı kullanıcı arayüzü.

Kullanılan Kütüphaneler
pygame: Grafik arayüz için.

chess: Satranç kuralları ve tahta yönetimi için.

numpy: Pozisyonel ödül hesaplamaları için.

🌟 Özellikler
Q-Öğrenme Optimizasyonu: Epsilon decay, materyal denge ve pozisyonel ödüllerle öğrenme.

Agresif Siyah Oyuncu: Taş alma ve şah çekme davranışları optimize edilmiştir.

Gerçek Zamanlı GUI: Hamleler anında görselleştirilir.

Oyun Sonu Tespiti: Mat, pat ve yetersiz materyal durumları desteklenir.

🤝 Katkı
Bu depoyu fork edin.

Yeni bir branch oluşturun:

bash
Copy
git checkout -b yeni-özellik
Değişikliklerinizi commit edin:

bash
Copy
git commit -m "Yeni özellik eklendi"
Değişikliklerinizi push edin:

bash
Copy
git push origin yeni-özellik
Pull Request oluşturun.

📜 Lisans
Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için LICENSE dosyasını inceleyin.
