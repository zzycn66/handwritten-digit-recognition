# mnist_dual_model_gui_v5.py
import os

from tensorflow import timestamp

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 可选：如果想强制CPU运行，取消注释

import sys, io, time, pickle, random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFrame, QMessageBox, QProgressBar, QFileDialog, QCheckBox, QLineEdit, QComboBox, QInputDialog
)
from PyQt5.QtGui import QPainter, QPixmap, QPen, QColor, QImage
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal
from PIL import Image, ImageOps
import numpy as np

MODEL_CNN = "mnist_cnn.h5"
MODEL_SVM = "mnist_svm.pkl"


# ========== 绘图画布 ==========
class PaintCanvas(QLabel):
    def __init__(self, parent=None, pen_width=22, size=320):
        super().__init__(parent)
        self.size_px = size
        self.setFixedSize(self.size_px, self.size_px)
        self.pix = QPixmap(self.size_px, self.size_px)
        self.pix.fill(Qt.white)
        self.setPixmap(self.pix)
        self.drawing = False
        self.last_point = QPoint()
        self.pen_width = pen_width
        self.pen_color = QColor(0, 0, 0)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)

    def set_pen_width(self, w):
        self.pen_width = w

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = e.pos()
            self._draw_point(self.last_point)

    def mouseMoveEvent(self, e):
        if self.drawing:
            self._draw_line(e.pos())

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = False

    def _draw_point(self, pt):
        p = QPainter(self.pix)
        pen = QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.drawPoint(pt)
        p.end()
        self.setPixmap(self.pix)

    def _draw_line(self, pos):
        p = QPainter(self.pix)
        pen = QPen(self.pen_color, self.pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.drawLine(self.last_point, pos)
        p.end()
        self.last_point = QPoint(pos)
        self.setPixmap(self.pix)

    def clear(self):
        self.pix.fill(Qt.white)
        self.setPixmap(self.pix)

    def get_image_pil(self):
        from PyQt5.QtCore import QBuffer, QIODevice
        buf = QBuffer()
        buf.open(QIODevice.ReadWrite)
        self.pix.save(buf, "PNG")
        return Image.open(io.BytesIO(buf.data()))

    # 修改：使用 QImage 而不是 PIL 的 toqpixmap
    def set_image_pil(self, pil_image):
        # 将 PIL 图像转换为 RGB 模式（如果需要）
        pil_image = pil_image.convert("RGB")
        # 获取图像数据
        data = pil_image.tobytes("raw", "RGB")
        # 创建 QImage
        # 参数: bytes_data, width, height, bytes_per_line, format
        qimg = QImage(data, pil_image.width, pil_image.height, pil_image.width * 3, QImage.Format_RGB888)
        # 创建 QPixmap 并设置到画布
        pixmap = QPixmap.fromImage(qimg)
        self.pix = pixmap
        self.setPixmap(self.pix)


# ========== 图像预处理 ==========
def preprocess_pil_image(pil, flatten=False):
    pil = pil.convert('L')
    img = np.array(pil)
    img = 255 - img
    img = (img > 50) * 255
    coords = np.column_stack(np.where(img > 0))
    if coords.size == 0:
        img = Image.new('L', (28, 28), 0)
        img_array = np.array(img).astype('float32') / 255.0
        return img_array.reshape(1, -1) if flatten else img_array.reshape(1, 28, 28, 1)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    img_cropped = img[y0:y1 + 1, x0:x1 + 1]
    img_pil = Image.fromarray(img_cropped)
    img_pil = ImageOps.fit(img_pil, (28, 28), Image.LANCZOS)
    img_array = np.array(img_pil).astype('float32') / 255.0

    if flatten:
        return img_array.reshape(1, -1)
    else:
        return img_array.reshape(1, 28, 28, 1)


# ========== CNN 训练线程 ==========
class CNNTrainThread(QThread):
    progress = pyqtSignal(str)
    done = pyqtSignal(float, float)
    error = pyqtSignal(str)

    def __init__(self, epochs=3):
        super().__init__()
        self.epochs = epochs

    def run(self):
        try:
            import tensorflow as tf
            from tensorflow.keras.utils import to_categorical
            from tensorflow.keras.datasets import mnist
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

            print("📦 [CNN] 加载 MNIST 数据...")
            self.progress.emit("加载 MNIST 数据...")
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)
            y_train = to_categorical(y_train, 10)
            y_test = to_categorical(y_test, 10)

            model = Sequential([
                Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                MaxPooling2D((2, 2)),
                Conv2D(32, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(64, activation='relu'),
                Dropout(0.25),
                Dense(10, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            start = time.time()
            for e in range(self.epochs):
                print(f"🚀 [CNN] 第 {e + 1}/{self.epochs} 轮训练中...")
                self.progress.emit(f"CNN 训练第 {e + 1}/{self.epochs} 轮...")
                model.fit(x_train, y_train, batch_size=128, epochs=1, validation_split=0.1, verbose=1)
            loss, acc = model.evaluate(x_test, y_test, verbose=0)
            elapsed = time.time() - start
            model.save(MODEL_CNN)
            self.done.emit(acc, elapsed)
        except Exception as e:
            self.error.emit(str(e))
            print("❌ [CNN] 错误:", e)


# ========== SVM 训练线程 ==========
class SVMTrainThread(QThread):
    progress = pyqtSignal(str)
    done = pyqtSignal(float, float)
    error = pyqtSignal(str)

    def run(self):
        try:
            from sklearn import svm, metrics
            from skimage.feature import hog
            from tensorflow.keras.datasets import mnist

            self.progress.emit("加载 MNIST 数据并提取 HOG 特征...")
            print("📦 [SVM] 正在加载 MNIST 数据并提取特征...")
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            def extract_hog(images):
                feats = []
                for i, img in enumerate(images):
                    feats.append(hog(img, orientations=9, pixels_per_cell=(4, 4),
                                     cells_per_block=(2, 2), block_norm='L2-Hys'))
                    if i % 20 == 0:
                        print(f"🌀 [SVM] 特征提取进度: {i}/{len(images)}")
                return np.array(feats)

            start = time.time()
            X_train = extract_hog(x_train[:20000])
            X_test = extract_hog(x_test[:5000])
            y_train = y_train[:20000]
            y_test = y_test[:5000]

            self.progress.emit("训练 SVM 模型中...")
            print("🚀 [SVM] 开始训练...")
            clf = svm.SVC(kernel='linear', probability=True)
            clf.fit(X_train, y_train)

            acc = metrics.accuracy_score(y_test, clf.predict(X_test))
            elapsed = time.time() - start
            with open(MODEL_SVM, "wb") as f:
                pickle.dump(clf, f)
            self.done.emit(acc, elapsed)
        except Exception as e:
            self.error.emit(str(e))
            print("❌ [SVM] 错误:", e)


# ========== CNN再训练线程 ==========
class CNNReTrainThread(QThread):
    progress = pyqtSignal(str)
    done = pyqtSignal(float)  # 只返回新准确率，因为训练很快
    error = pyqtSignal(str)

    def __init__(self, feedback_images, feedback_labels, epochs=1):
        super().__init__()
        self.feedback_images = feedback_images
        self.feedback_labels = feedback_labels
        self.epochs = epochs

    def run(self):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            from tensorflow.keras.utils import to_categorical

            print(f"🔄 [CNN] 加载现有模型进行再训练...")
            self.progress.emit("加载现有模型...")
            model = load_model(MODEL_CNN)

            # 预处理反馈图像
            # preprocess_pil_image 返回 (1, 28, 28, 1) 形状
            processed_image_list = []
            for img in self.feedback_images:
                processed_img = preprocess_pil_image(img, flatten=False)  # 得到 (1, 28, 28, 1)
                processed_image_list.append(processed_img[0])  # 取出 (28, 28, 1) 部分
            # 堆叠成 (num_samples, 28, 28, 1)
            processed_images = np.stack(processed_image_list, axis=0)
            processed_labels = to_categorical(self.feedback_labels, 10)

            print(f"🔄 [CNN] 使用 {len(processed_images)} 个反馈样本进行再训练...")
            print(f"Input shape: {processed_images.shape}, Label shape: {processed_labels.shape}")
            self.progress.emit(f"使用 {len(processed_images)} 个反馈样本再训练 {self.epochs} 轮...")

            # 进行增量训练
            model.fit(processed_images, processed_labels, batch_size=32, epochs=self.epochs, verbose=1)

            # 保存更新后的模型
            model.save(MODEL_CNN)
            print("✅ [CNN] 再训练完成，模型已更新。")
            self.progress.emit("再训练完成，模型已更新。")
            # 这里可以简单评估，但通常我们会用一个固定的验证集来评估
            # 假设用MNIST测试集评估
            (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_test = x_test.astype('float32') / 255.0
            x_test = np.expand_dims(x_test, -1)
            y_test_cat = to_categorical(y_test, 10)
            _, acc = model.evaluate(x_test, y_test_cat, verbose=0)
            self.done.emit(acc)

        except Exception as e:
            self.error.emit(str(e))
            print("❌ [CNN] 再训练错误:", e)


# ========== MNIST数据加载线程 ==========
class MNISTLoadThread(QThread):
    data_loaded = pyqtSignal(object, object)  # 信号：发送加载完成的数据
    error = pyqtSignal(str)  # 信号：发送错误信息

    def run(self):
        try:
            print("📦 后台加载MNIST测试集...")
            from tensorflow.keras.datasets import mnist
            (_, _), (x_test, y_test) = mnist.load_data()
            # 发送加载完成的数据给主线程
            self.data_loaded.emit(x_test, y_test)
            print("✅ MNIST测试集加载完成")
        except Exception as e:
            # 发送错误信息给主线程
            self.error.emit(str(e))


# ========== 主窗口 ==========
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手写数字识别")
        self.setMinimumSize(950, 550)
        self.model = None
        self.model_type = "CNN"
        # 修改：移除直接存储数据的变量，改为存储线程引用
        self.mnist_thread = None
        self.mnist_test_images = None
        self.mnist_test_labels = None
        # 新增：存储反馈数据
        self.feedback_data = []  # List of (PIL_image, label)
        self._init_ui()

    def _init_ui(self):
        c = QWidget()
        self.setCentralWidget(c)
        layout = QHBoxLayout(c)
        left = QVBoxLayout()
        right = QVBoxLayout()
        layout.addLayout(left, 2)
        layout.addLayout(right, 1)

        self.canvas = PaintCanvas(size=360, pen_width=20)
        left.addWidget(self.canvas, alignment=Qt.AlignCenter)

        # 按钮行：清除、保存、随机MNIST
        btn_row = QHBoxLayout()
        for text, color, fn in [("清除", "#ef5350", self.canvas.clear),
                                ("保存", "#42a5f5", self._save),
                                ("随机MNIST", "#9ccc65", self._load_random_mnist)]:  # 添加新按钮
            b = QPushButton(text)
            b.setStyleSheet(f"background:{color};color:white;border:none;border-radius:6px;")
            b.setFixedHeight(36)
            b.clicked.connect(fn)
            btn_row.addWidget(b)
        left.addLayout(btn_row)

        # 模型选择与加载
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("识别方式："))
        self.model_selector = QComboBox()
        self.model_selector.addItems(["CNN", "HOG+SVM"])
        self.model_selector.currentTextChanged.connect(self._change_model_type)
        model_row.addWidget(self.model_selector)

        self.load_btn = QPushButton("加载模型")
        self.load_btn.clicked.connect(self._load_model)
        model_row.addWidget(self.load_btn)
        left.addLayout(model_row)

        # 训练控制
        train_row = QHBoxLayout()
        self.train_btn = QPushButton("训练模型")
        self.train_btn.clicked.connect(self._train)
        self.epoch_input = QLineEdit("3")
        self.epoch_input.setFixedWidth(60)
        train_row.addWidget(self.train_btn)
        train_row.addWidget(QLabel("轮数："))
        train_row.addWidget(self.epoch_input)
        left.addLayout(train_row)

        # 新增：再训练按钮 (仅对CNN有效)
        self.retrain_btn = QPushButton("使用反馈数据再训练 (CNN)")
        self.retrain_btn.clicked.connect(self._retrain_with_feedback)
        left.addWidget(self.retrain_btn)

        # 右侧：结果和概率分布
        self.status = QLabel("状态：未加载模型")
        right.addWidget(self.status)
        self.pred_label = QLabel("-")
        self.pred_label.setStyleSheet("font-size:72px; font-weight:bold;")
        right.addWidget(self.pred_label)

        self.bars = []
        for i in range(10):
            h = QHBoxLayout()
            l = QLabel(f"{i}:")
            p = QProgressBar()
            p.setRange(0, 1000)
            h.addWidget(l)
            h.addWidget(p)
            right.addLayout(h)
            self.bars.append(p)

        # 预测和反馈按钮行
        pred_btn_row = QHBoxLayout()
        self.predict_btn = QPushButton("识别")
        self.predict_btn.clicked.connect(self._predict)
        pred_btn_row.addWidget(self.predict_btn)

        self.feedback_btn = QPushButton("反馈")
        self.feedback_btn.clicked.connect(self._request_feedback)
        pred_btn_row.addWidget(self.feedback_btn)
        right.addLayout(pred_btn_row)

    def _change_model_type(self, text):
        self.model_type = text
        self.model = None
        self._set_status(f"切换为 {text} 模式")

    def _set_status(self, msg, err=False):
        self.status.setText(("❌ " if err else "✅ ") + msg)

    def _train(self):
        if self.model_type == "CNN":
            try:
                epochs = int(self.epoch_input.text())
            except:
                QMessageBox.warning(self, "错误", "请输入整数轮数")
                return
            print(f"\n==================== 开始 CNN 训练 ({epochs} epoch) ====================")
            self.thread = CNNTrainThread(epochs)
        else:
            print("\n==================== 开始 HOG+SVM 训练 ====================")
            self.thread = SVMTrainThread()

        self.thread.progress.connect(lambda s: self._set_status(s))
        self.thread.done.connect(self._on_train_done)
        self.thread.error.connect(lambda e: self._set_status(e, True))
        self.thread.start()

    def _on_train_done(self, acc, elapsed):
        self._set_status(f"{self.model_type} 训练完成 acc={acc:.4f}，耗时 {elapsed:.1f} 秒")
        QMessageBox.information(self, "训练完成",
                                f"{self.model_type} 模型训练完成！\n准确率：{acc:.4f}\n耗时：{elapsed:.1f} 秒")
        self.model = None

    def _load_model(self):
        try:
            if self.model_type == "CNN":
                from tensorflow.keras.models import load_model
                self.model = load_model(MODEL_CNN)
            else:
                with open(MODEL_SVM, "rb") as f:
                    self.model = pickle.load(f)
            self._set_status(f"{self.model_type} 模型已加载")
        except Exception as e:
            self._set_status(f"加载失败：{e}", True)

    def _predict(self):
        pil = self.canvas.get_image_pil()
        if self.model is None:
            QMessageBox.warning(self, "错误", "请先加载或训练模型！")
            return

        if self.model_type == "CNN":
            x = preprocess_pil_image(pil)
            preds = self.model.predict(x, verbose=0)[0]
        else:
            from skimage.feature import hog
            img = preprocess_pil_image(pil, flatten=False).reshape(28, 28)
            feat = hog(img, orientations=9, pixels_per_cell=(4, 4),
                       cells_per_block=(2, 2), block_norm='L2-Hys').reshape(1, -1)
            preds = self.model.predict_proba(feat)[0]

        idx = int(np.argmax(preds))
        self.pred_label.setText(str(idx))

        for i, p in enumerate(preds):
            self.bars[i].setValue(int(p * 1000))
            self.bars[i].setFormat(f"{p * 100:.1f}%")

        self._set_status(f"{self.model_type} 预测完成：{idx}")

    def _save(self):
        timestamp = int(time.time())
        fname, _ = QFileDialog.getSaveFileName(self, "保存图像", f"digit_{timestamp}.png", "PNG Files (*.png)")
        if fname:
            self.canvas.get_image_pil().save(fname)

    def _load_random_mnist(self):
        """加载随机MNIST图片到画布并进行识别"""
        # 如果数据已经加载，则直接处理
        if self.mnist_test_images is not None and self.mnist_test_labels is not None:
            print("🔄 使用已缓存的MNIST数据")
            self._process_random_mnist()
            return

        # 如果数据未加载且没有正在进行的加载线程，则启动新线程
        if self.mnist_thread is None or not self.mnist_thread.isRunning():
            print("🔄 启动MNIST数据加载线程")
            self.mnist_thread = MNISTLoadThread()
            # 连接线程的信号
            self.mnist_thread.data_loaded.connect(self._on_mnist_loaded)
            self.mnist_thread.error.connect(lambda e: self._set_status(f"加载MNIST数据集失败：{e}", True))
            self.mnist_thread.start()
            self._set_status("正在加载MNIST数据集...")
        else:
            # 如果线程正在运行，可以提示用户稍等
            self._set_status("MNIST数据集加载中，请稍候...")

    def _on_mnist_loaded(self, x_test, y_test):
        """接收MNIST加载线程完成后的数据"""
        # 将加载的数据存储到实例变量中
        self.mnist_test_images = x_test
        self.mnist_test_labels = y_test
        # 数据加载完成后，处理随机图片
        self._process_random_mnist()

    def _process_random_mnist(self):
        """处理随机选择的MNIST图片（在数据已加载后调用）"""
        # 随机选择一张图片
        random_index = random.randint(0, len(self.mnist_test_images) - 1)
        selected_image_array = self.mnist_test_images[random_index]
        true_label = int(self.mnist_test_labels[random_index])

        # --- 关键修改：直接对原始MNIST数组进行与训练时相同的预处理 ---
        # 1. 转换为float32并归一化
        x = selected_image_array.astype('float32') / 255.0
        # 2. 添加通道维度 (28, 28) -> (28, 28, 1)
        x = np.expand_dims(x, -1)
        # 3. 添加批次维度 (28, 28, 1) -> (1, 28, 28, 1)
        x = np.expand_dims(x, 0)
        # 预处理完成，x 的形状现在是 (1, 28, 28, 1)，与训练时一致

        # --- 保持原始图像用于显示（可选） ---
        # 将原始MNIST数组转换为PIL图像用于显示
        pil_img = Image.fromarray(selected_image_array.astype('uint8'))
        pil_img_rgb = pil_img.convert("RGB")
        # 将图片尺寸调整到画布大小 (360x360)
        resized_img = pil_img_rgb.resize((self.canvas.size_px, self.canvas.size_px), Image.LANCZOS)
        # 设置到画布上
        self.canvas.set_image_pil(resized_img)

        # --- 使用预处理后的数据进行预测 ---
        if self.model is not None and self.model_type == "CNN":
            try:
                # 直接使用预处理好的 x 进行预测
                preds = self.model.predict(x, verbose=0)[0]  # verbose=0 避免打印进度
                idx = int(np.argmax(preds))

                # --- 更新UI ---
                # 显示预测结果
                self.pred_label.setText(str(idx))
                # 更新概率条
                for i, p in enumerate(preds):
                    self.bars[i].setValue(int(p * 1000))
                    self.bars[i].setFormat(f"{p * 100:.1f}%")

                # 在状态栏显示真实标签和预测结果
                status_msg = f"随机MNIST图片加载完成，真实标签: {true_label}，模型预测: {idx}"
                if idx == true_label:
                    self._set_status(status_msg + " (正确)")
                else:
                    self._set_status(status_msg + " (错误)", err=True)
            except Exception as e:
                self._set_status(f"预测时发生错误: {e}", err=True)
        elif self.model is not None and self.model_type == "HOG+SVM":
            # SVM 使用 HOG 特征，需要专门的预处理
            from skimage.feature import hog
            # 注意：这里使用的预处理也应与SVM训练时一致
            # x 是 (1, 28, 28, 1)，取第一个样本并移除通道维度得到 (28, 28)
            img_for_hog = x[0].reshape(28, 28)
            feat = hog(img_for_hog, orientations=9, pixels_per_cell=(4, 4),
                       cells_per_block=(2, 2), block_norm='L2-Hys').reshape(1, -1)
            try:
                preds = self.model.predict_proba(feat)[0]
                idx = int(np.argmax(preds))

                # --- 更新UI ---
                self.pred_label.setText(str(idx))
                for i, p in enumerate(preds):
                    self.bars[i].setValue(int(p * 1000))
                    self.bars[i].setFormat(f"{p * 100:.1f}%")

                status_msg = f"随机MNIST图片加载完成，真实标签: {true_label}，模型预测: {idx}"
                if idx == true_label:
                    self._set_status(status_msg + " (正确)")
                else:
                    self._set_status(status_msg + " (错误)", err=True)
            except Exception as e:
                self._set_status(f"预测时发生错误: {e}", err=True)
        else:
            # 如果模型未加载
            self._set_status(f"随机MNIST图片加载完成，真实标签: {true_label} (请先加载模型进行预测)")
            # 仍然将图像显示在画布上

    def _request_feedback(self):
        """请求用户输入正确的标签"""
        if self.model is None:
            QMessageBox.warning(self, "错误", "请先加载或训练模型！")
            return

        # 获取当前画布图像
        current_image = self.canvas.get_image_pil()
        if current_image is None:
            QMessageBox.warning(self, "错误", "画布为空，无法反馈。")
            return

        # 弹出输入对话框
        correct_label, ok = QInputDialog.getInt(self, "反馈", "识别错误，请输入正确的数字 (0-9):", 0, 0, 9)
        if ok:
            # 将图像和用户输入的标签添加到反馈列表
            self.feedback_data.append((current_image, correct_label))
            self._set_status(f"反馈已记录: 标签 {correct_label}。当前反馈池大小: {len(self.feedback_data)}")

    def _retrain_with_feedback(self):
        """使用反馈数据对CNN模型进行再训练"""
        if self.model_type != "CNN":
            QMessageBox.warning(self, "错误", "再训练功能仅适用于CNN模型。")
            return

        if not self.feedback_data:
            QMessageBox.information(self, "信息", "没有反馈数据可用于再训练。")
            return

        # 检查模型是否存在
        if not os.path.exists(MODEL_CNN):
            QMessageBox.warning(self, "错误", f"模型文件 {MODEL_CNN} 不存在，请先训练模型。")
            return

        print(f"\n==================== 开始使用 {len(self.feedback_data)} 个反馈样本进行CNN再训练 ====================")
        # 提取图像和标签
        images, labels = zip(*self.feedback_data)
        self.retrain_thread = CNNReTrainThread(list(images), list(labels))
        self.retrain_thread.progress.connect(lambda s: self._set_status(s))
        self.retrain_thread.done.connect(self._on_retrain_done)
        self.retrain_thread.error.connect(lambda e: self._set_status(f"再训练失败: {e}", True))
        self.retrain_thread.start()

    def _on_retrain_done(self, new_acc):
        self._set_status(f"CNN 模型使用反馈数据再训练完成！新的测试准确率: {new_acc:.4f}")
        QMessageBox.information(self, "再训练完成",
                                f"CNN 模型已使用反馈数据更新！\n新的测试准确率: {new_acc:.4f}")
        # 重新加载更新后的模型
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(MODEL_CNN)
            self._set_status(f"{self.model_type} 模型已重新加载 (使用反馈数据更新)")
        except Exception as e:
            self._set_status(f"重新加载模型失败：{e}", True)
        # 清空反馈池
        self.feedback_data = []
        self._set_status(f"反馈池已清空。")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()