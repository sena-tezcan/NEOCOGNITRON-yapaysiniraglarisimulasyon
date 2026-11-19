import numpy as np
from scipy.signal import correlate2d
from scipy.ndimage import rotate
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def generate_synthetic_digits(n_samples=600, shift_px=0, seed=42):
    """
    28x28 boyutunda, ±10 derece eğimli ve farklı yazı tiplerine sahip
    sentetik rakam (0-9) veri seti üretir.
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, 28, 28), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)

    font_paths = [
        "arial.ttf",
        "times.ttf",
        "cour.ttf",
        "verdana.ttf",
    ]
    fonts = []
    for fp in font_paths:
        try:
            fonts.append(ImageFont.truetype(fp, size=20))
        except IOError:
            pass
    if not fonts:
        fonts = [ImageFont.load_default()]

    for i in range(n_samples):
        digit = i % 10
        y[i] = digit
        font = fonts[rng.integers(0, len(fonts))]
        angle = rng.uniform(-10.0, 10.0)

        img = Image.new("L", (28, 28), color=0)
        draw = ImageDraw.Draw(img)
        draw.text((8, 3), str(digit), fill=255, font=font)

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = rotate(arr, angle, reshape=False, order=1)

        # Yatay kaydırma (0-8 px)
        if shift_px > 0:
            shifted = np.zeros_like(arr)
            if shift_px < 28:
                shifted[:, shift_px:] = arr[:, :-shift_px]
            arr = shifted

        X[i] = np.clip(arr, 0.0, 1.0)

    return X, y


def get_4_direction_kernels():
    """1. Adım (US1) için 4 yönlü sabit çizgi çekirdekleri (yatay, dikey, 2 çapraz)."""
    k_horiz = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
    k_vert = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)
    k_diag1 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32)
    k_diag2 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)
    kernels = [k_horiz, k_vert, k_diag1, k_diag2]
    return [k / (np.linalg.norm(k) + 1e-8) for k in kernels]


def maxpool_3x3_stride2(feat_map):
    """2. Adım (UC1): 3x3 max-havuzlama (adım = 2)."""
    C, H, W = feat_map.shape
    out_h = (H - 3) // 2 + 1
    out_w = (W - 3) // 2 + 1
    pooled = np.zeros((C, out_h, out_w), dtype=np.float32)
    for c in range(C):
        for i in range(out_h):
            for j in range(out_w):
                patch = feat_map[c, i * 2 : i * 2 + 3, j * 2 : j * 2 + 3]
                pooled[c, i, j] = np.max(patch)
    return pooled


def extract_patches(c1, patch_size=5):
    """UC1 çıktısından US2 katmanı için yerel yamalar çıkarır."""
    C, H, W = c1.shape
    patches = []
    for i in range(H - patch_size + 1):
        for j in range(W - patch_size + 1):
            patch = c1[:, i : i + patch_size, j : j + patch_size].reshape(-1)
            patches.append(patch)
    return np.array(patches, dtype=np.float32)  # (N_patches, C * patch_size^2)


class SimplifiedNeocognitron:
    def __init__(self, n_planes=60, theta=0.5, lr=0.05, seed=42):
        self.n_planes = n_planes
        self.theta = theta
        self.lr = lr
        self.kernels_us1 = get_4_direction_kernels()
        rng = np.random.default_rng(seed)
        # 4 kanal x 5x5 yama = 100 boyutlu ağırlık vektörleri (60 düzlem)
        W = rng.uniform(0.0, 1.0, size=(n_planes, 4 * 5 * 5)).astype(np.float32)
        self.W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)

    def forward_us1_uc1(self, img):
        # 1. US1: 4 yönlü sabit çizgi çekirdeği
        s1 = np.stack(
            [np.maximum(0.0, correlate2d(img, k, mode="same")) for k in self.kernels_us1],
            axis=0,
        )
        # 2. UC1: 3x3 max-havuzlama (adım = 2)
        c1 = maxpool_3x3_stride2(s1)
        return c1

    def train_unsupervised_us2(self, X_train, epochs=3):
        """3. Adım (US2): 60 düzlem için kazanan-hepsini-alır denetimsiz öğrenme."""
        for _ in range(epochs):
            for img in X_train:
                c1 = self.forward_us1_uc1(img)
                patches = extract_patches(c1, patch_size=5)
                norms = np.linalg.norm(patches, axis=1)
                valid = patches[norms > 1e-6]
                for x in valid:
                    x_norm = x / (np.linalg.norm(x) + 1e-8)
                    k = np.argmax(self.W @ x_norm)  # kazanan düzlem
                    self.W[k] += self.lr * (x_norm - self.W[k])  # sinaps güncelleme
                    self.W[k] /= np.linalg.norm(self.W[k]) + 1e-8

    def extract_uc2_features(self, X):
        """4. Adım (UC2): Global max ile konumdan bağımsız 60 öznitelik."""
        features = np.zeros((len(X), self.n_planes), dtype=np.float32)
        for idx, img in enumerate(X):
            c1 = self.forward_us1_uc1(img)
            patches = extract_patches(c1, patch_size=5)  # (N_patches, 100)
            p_norm = np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8
            w_norm = np.linalg.norm(self.W, axis=1, keepdims=True).T + 1e-8
            cos = (patches @ self.W.T) / (p_norm * w_norm)  # (N_patches, 60)
            s2 = np.maximum(0.0, cos - self.theta) / (1.0 - self.theta)
            c2 = s2.max(axis=0)  # tüm uzamsal konumlar üzerinden global max (60 sayı)
            features[idx] = c2
        return features


if __name__ == "__main__":
    # 1. Sadece merkezde (0 px kayma) 600 eğitim verisi üret
    X_train, y_train = generate_synthetic_digits(n_samples=600, shift_px=0, seed=42)

    # 2. Neocognitron modelini oluştur ve US2 katmanını denetimsiz eğit
    model = SimplifiedNeocognitron(n_planes=60, theta=0.5, lr=0.05, seed=42)
    model.train_unsupervised_us2(X_train, epochs=3)

    # 3. 5. Adım: UC2'den çıkan 60 öznitelik ile Lojistik Regresyon sınıflandırıcıyı eğit
    F_train = model.extract_uc2_features(X_train)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(F_train, y_train)

    # 4. 0-8 px yatay kaydırma testi (300 test örneği)
    for shift in [0, 3, 6, 8]:
        X_test, y_test = generate_synthetic_digits(n_samples=300, shift_px=shift, seed=99)
        F_test = model.extract_uc2_features(X_test)
        preds = clf.predict(F_test)
        acc = accuracy_score(y_test, preds) * 100
        print(f"Kayma {shift} px -> Doğruluk: %{acc:.1f}")
