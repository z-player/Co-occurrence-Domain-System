"""
共现域系统 (CDS) 概念验证 · 教学完整版
- 前端(器官层): 3x3最大池化降采样 + 二值化 (将物理像素映射为元坐标)
- 编码空间: 9x9 网格，每个激活像素为一个元（坐标占位）
- 学习: 正向闭包生成 1~N 阶共现域，倒排索引加速
- 推理: 最高阶共现域子集包含投票，O(1)寻址
- 记忆管理: 全局降阶 (所有共现域阶数-1，一阶转为元或删除)
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional
import hashlib
from torchvision import datasets, transforms

# ============================================================================
# 数据加载与预处理 (模拟器官层：物理信号 → 元坐标)
# ============================================================================

def downsample_maxpool(img: np.ndarray, pool_size: int = 3) -> np.ndarray:
    """最大池化降采样，保持二值特性（类似视网膜神经节细胞的感受野）"""
    h, w = img.shape
    new_h, new_w = h // pool_size, w // pool_size
    img = img[:new_h * pool_size, :new_w * pool_size]
    img_reshaped = img.reshape(new_h, pool_size, new_w, pool_size)
    down = img_reshaped.max(axis=(1, 3))
    return down

def load_mnist_subset(digits: List[int], samples_per_digit: int,
                      train: bool = True, data_dir: str = './data') -> Tuple[np.ndarray, np.ndarray]:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    targets = dataset.targets.numpy() if hasattr(dataset.targets, 'numpy') else np.array(dataset.targets)
    images, labels = [], []
    for d in digits:
        idx = np.where(targets == d)[0][:samples_per_digit]
        for i in idx:
            img_tensor, label = dataset[i]
            img = img_tensor.squeeze().numpy()
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

def preprocess_image(img: np.ndarray, pool_size: int = 3, threshold: float = 0.5) -> np.ndarray:
    """
    器官前端处理：
    1. 池化降维 (空间压缩)
    2. 二值化 (将像素值映射为元：激活=1，抑制=不占位)
    输出为 0/1 矩阵，每个激活点即为一个元坐标。
    """
    down = downsample_maxpool(img, pool_size)
    binary = (down > threshold).astype(np.uint8)
    return binary

# ============================================================================
# 共现域哈希 (用于快速唯一标识)
# ============================================================================

def binary_matrix_to_bytes(mat: np.ndarray) -> bytes:
    return mat.tobytes()

def compute_codomain_id(mat: np.ndarray) -> str:
    """根据矩阵内容生成唯一ID (共现域的坐标集指纹)"""
    return hashlib.md5(binary_matrix_to_bytes(mat)).hexdigest()[:8]

# ============================================================================
# 核心系统: 共现域系统 (CDS)
# ============================================================================

class CooccurrenceDomainSystem:
    def __init__(self):
        # 原始图像记录 (用于追溯，推理时不依赖)
        self.images: Dict[int, np.ndarray] = {}
        self.img_labels: Dict[int, int] = {}
        self.next_img_id = 0

        # 共现域存储: cid -> {matrix, order, img_ids}
        self.codomains: Dict[str, dict] = {}

        # 倒排索引 (图像→共现域 & 共现域→图像)
        self.inv_index: Dict[str, Set[int]] = defaultdict(set)   # cid -> set of img_ids
        self.img_to_codomains: Dict[int, Set[str]] = defaultdict(set)

        self.total_codomains_created = 0

        # 元坐标→共现域的倒排索引 (推理引擎关键结构)
        self.pixel_to_codomains: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        self.index_built = False

    # -------------------- 索引构建 --------------------
    def build_pixel_index(self):
        """构建元坐标→共现域的倒排索引，实现O(1)候选召回"""
        self.pixel_to_codomains.clear()
        for cid, info in self.codomains.items():
            ys, xs = np.where(info['matrix'] == 1)
            for y, x in zip(ys, xs):
                self.pixel_to_codomains[(y, x)].add(cid)
        self.index_built = True

    # -------------------- 核心记忆操作 --------------------
    def _add_codomain(self, matrix: np.ndarray, order: int, img_ids: Set[int]) -> Optional[str]:
        """注册一个新共现域或更新已有共现域的关联图像"""
        if np.sum(matrix) == 0:
            return None
        cid = compute_codomain_id(matrix)
        if cid in self.codomains:
            # 已存在：增加证据图像
            self.codomains[cid]['img_ids'].update(img_ids)
            for iid in img_ids:
                self.inv_index[cid].add(iid)
                self.img_to_codomains[iid].add(cid)
            return cid

        # 新共现域
        self.codomains[cid] = {
            'matrix': matrix.copy(),
            'order': order,
            'img_ids': img_ids.copy()
        }
        self.total_codomains_created += 1
        for iid in img_ids:
            self.inv_index[cid].add(iid)
            self.img_to_codomains[iid].add(cid)
        if self.index_built:
            ys, xs = np.where(matrix == 1)
            for y, x in zip(ys, xs):
                self.pixel_to_codomains[(y, x)].add(cid)
        return cid

    def learn_image(self, img: np.ndarray, label: int) -> int:
        """
        正向闭包学习：
        1. 将输入图像注册为一阶共现域
        2. 与所有已有共现域求交，生成更高阶共现域
        """
        img_id = self.next_img_id
        self.next_img_id += 1
        self.images[img_id] = img.copy()
        self.img_labels[img_id] = label

        # 自身一阶共现域
        self._add_codomain(img, order=1, img_ids={img_id})

        # 与已有共现域求交，生成高阶 (正向闭包)
        existing = list(self.codomains.items())
        for cid, info in existing:
            if img_id in info['img_ids']:
                continue
            intersection = np.logical_and(info['matrix'], img).astype(np.uint8)
            if np.sum(intersection) == 0:
                continue
            new_order = info['order'] + 1
            new_ids = info['img_ids'].copy()
            new_ids.add(img_id)
            self._add_codomain(intersection, order=new_order, img_ids=new_ids)

        return img_id

    def fit(self, images: np.ndarray, labels: np.ndarray,
            decay_interval: int = 500, decay_start: Optional[int] = None):
        """
        批量学习，支持延迟开启的全局降阶（模拟记忆巩固）
        :param decay_interval: 降阶间隔（每学多少张图触发一次）
        :param decay_start: 从第几张开始降阶（None表示不降阶）
        """
        for i, (img, lbl) in enumerate(zip(images, labels)):
            self.learn_image(img, lbl)
            img_count = i + 1

            if decay_start is not None and img_count >= decay_start:
                if (img_count - decay_start) % decay_interval == 0:
                    self.decay_all_codomains()

        self.build_pixel_index()
        print(f"学习完成。共 {len(self.images)} 张图像，生成 {len(self.codomains)} 个共现域。")

    # -------------------- 推理引擎 --------------------
    def query_image_fast(self, img: np.ndarray) -> Set[str]:
        """快速子集包含：利用元坐标倒排索引召回候选，再严格验证"""
        if not self.index_built:
            self.build_pixel_index()

        candidates = set()
        ys, xs = np.where(img == 1)   # 当前输入的激活元坐标
        for y, x in zip(ys, xs):
            candidates.update(self.pixel_to_codomains.get((y, x), set()))

        activated = set()
        for cid in candidates:
            info = self.codomains[cid]
            # 子集包含检查：共现域的所有1位置在输入中必须为1
            if np.all(img[info['matrix'] == 1] == 1):
                activated.add(cid)
        return activated

    def classify_by_highest_order(self, img: np.ndarray) -> Tuple[int, Set[str]]:
        """最高阶推理：激活的共现域中，阶数最高者投票决定类别"""
        activated = self.query_image_fast(img)
        if not activated:
            return -1, activated

        max_order = max(self.codomains[cid]['order'] for cid in activated)
        top_codomains = [cid for cid in activated if self.codomains[cid]['order'] == max_order]

        label_votes = defaultdict(int)
        for cid in top_codomains:
            for iid in self.inv_index[cid]:
                label_votes[self.img_labels[iid]] += 1

        if not label_votes:
            return -1, activated

        best_label = max(label_votes, key=label_votes.get)
        return best_label, activated

    def evaluate(self, test_images: np.ndarray, test_labels: np.ndarray) -> float:
        correct = 0
        for img, true_label in zip(test_images, test_labels):
            pred_label, _ = self.classify_by_highest_order(img)
            if pred_label == true_label:
                correct += 1
        return correct / len(test_images)

    # -------------------- 记忆生态管理 (降阶遗忘) --------------------
    def decay_all_codomains(self):
        """
        全局降阶操作：
        - 删除所有1阶共现域（具体图像记忆），其元坐标保留在元空间中
        - 其余共现域阶数减1，模拟记忆抽象化
        """
        to_remove = [cid for cid, info in self.codomains.items() if info['order'] == 1]
        for cid in to_remove:
            self._remove_codomain(cid)
        for info in self.codomains.values():
            if info['order'] > 1:
                info['order'] -= 1
        print(f"降阶完成：删除一阶共现域 {len(to_remove)} 个，当前共现域总数: {len(self.codomains)}")

    def _remove_codomain(self, cid: str):
        """移除一个共现域，清理所有索引"""
        info = self.codomains.pop(cid)
        # 图像索引清理
        for iid in info['img_ids']:
            self.inv_index[cid].discard(iid)
            self.img_to_codomains[iid].discard(cid)
        # 元坐标索引清理
        if self.index_built:
            ys, xs = np.where(info['matrix'] == 1)
            for y, x in zip(ys, xs):
                self.pixel_to_codomains[(y, x)].discard(cid)

    def get_statistics(self):
        """获取当前记忆生态统计"""
        order_counts = defaultdict(int)
        for info in self.codomains.values():
            order_counts[info['order']] += 1
        return {
            'total_images': len(self.images),
            'total_codomains': len(self.codomains),
            'order_distribution': dict(order_counts),
            'max_order': max(order_counts.keys()) if order_counts else 0
        }

# ============================================================================
# 演示入口
# ============================================================================

def run_demo():
    print("=" * 70)
    print("共现域系统 (CDS) 概念验证 · 最高阶推理 + 降阶遗忘")
    print("=" * 70)

    digits = [0, 1]          # 二分类任务，便于观察高阶共现域
    train_per_class = 40
    test_per_class = 4

    print(f"\n加载 MNIST 子集: 数字 {digits}")
    print(f"训练: 每类 {train_per_class} 张, 测试: 每类 {test_per_class} 张")

    X_train, y_train = load_mnist_subset(digits, train_per_class, train=True)
    X_test, y_test = load_mnist_subset(digits, test_per_class, train=False)

    print("\n器官前端: 3x3池化 → 9x9 二值化 (阈值0.5)")
    X_train_proc = np.array([preprocess_image(img) for img in X_train])
    X_test_proc = np.array([preprocess_image(img) for img in X_test])
    print(f"处理后尺寸: {X_train_proc[0].shape}，每个激活像素对应一个元坐标")

    system = CooccurrenceDomainSystem()
    print("\n开始学习 (从第40张起，每14张降阶一次)...")
    system.fit(X_train_proc, y_train, decay_interval=14, decay_start=40)

    stats = system.get_statistics()
    print(f"\n记忆生态统计:")
    print(f"  训练图像: {stats['total_images']}")
    print(f"  共现域总数: {stats['total_codomains']}")
    print(f"  阶数分布: {stats['order_distribution']}")
    print(f"  最高阶数: {stats['max_order']}")

    print("\n推理测试 (仅最高阶共现域投票)...")
    train_acc = system.evaluate(X_train_proc, y_train)
    test_acc = system.evaluate(X_test_proc, y_test)

    print(f"\n准确率:  训练集 {train_acc:.2%} ｜ 测试集 {test_acc:.2%}")

    # 单样本推理细节
    print("\n" + "-" * 50)
    sample_img = X_test_proc[0]
    sample_label = y_test[0]
    pred_label, activated = system.classify_by_highest_order(sample_img)

    print(f"测试样本 真实: {sample_label}, 预测: {pred_label}")
    print(f"激活的共现域数量: {len(activated)}")
    if activated:
        max_order = max(system.codomains[cid]['order'] for cid in activated)
        top = [cid for cid in activated if system.codomains[cid]['order'] == max_order]
        print(f"最高阶: {max_order}, 该阶共现域数: {len(top)}")
        print(f"样本支持度: {[len(system.codomains[cid]['img_ids']) for cid in top[:3]]}")

    print("\n" + "=" * 70)
    print("演示结束。")
    print("=" * 70)

if __name__ == "__main__":
    run_demo()