import pandas as pd
import numpy as np
import scorecardpy as sc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ScorecardModel:
    def __init__(self):
        self.bins = None
        self.model = None
        self.card = None

    def load_data(self, data_path):
        """加载数据并进行基本预处理"""
        try:
            data = pd.read_csv(data_path)
            return data
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None

    def preprocess_data(self, data, target, exclude_cols=None):
        """数据预处理和特征筛选"""
        if exclude_cols is None:
            exclude_cols = []
        
        # 排除不需要的列
        feature_cols = [col for col in data.columns if col not in exclude_cols + [target]]
        
        # 特征分箱
        self.bins = sc.woebin(data, y=target, x=feature_cols)
        
        # WOE转换
        woe_data = sc.woebin_ply(data, self.bins)
        
        # 获取转换后的列名（确保与原始特征列名一致）
        woe_cols = [col for col in woe_data.columns if col.endswith('_woe')]
        
        # 确保目标列名正确
        if target not in woe_data.columns:
            woe_data[target] = data[target]
        
        # 调试信息
        print(f"原始数据列名: {data.columns}")
        print(f"WOE转换后列名: {woe_data.columns}")
        print(f"最终使用的WOE列: {woe_cols}")
        print(f"目标列: {target}")
        
        return woe_data, woe_cols

    def train_model(self, woe_data, target):
        """训练评分卡模型"""
        try:
            # 获取WOE转换后的列名
            woe_cols = [col for col in woe_data.columns if col.endswith('_woe')]
            
            # 确保使用正确的列名
            if not all(col in woe_data.columns for col in woe_cols + [target]):
                print(f"可用列: {woe_data.columns}")
                print(f"需要列: {woe_cols + [target]}")
                raise ValueError("WOE转换后的列名不匹配")
            
            # 分割训练集和测试集
            X = woe_data[woe_cols]
            y = woe_data[target]
            
            # 打印调试信息
            print(f"训练集特征列: {X.columns}")
            print(f"训练集目标列: {y.name}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

            # 训练模型
            print("[INFO] 开始训练评分卡模型...")
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(random_state=42)
            lr.fit(X_train, y_train)
            
            # 生成评分卡
            print("[INFO] 生成评分卡...")
            # 使用WOE转换后的列名
            # 使用WOE转换后的列名
            # 使用WOE转换后的列名
            self.model = sc.scorecard(self.bins, lr, woe_cols)
            
            # 生成评分卡
            # 使用WOE转换后的列名
            self.card = sc.scorecard_ply(woe_data.rename(columns={
                'income_woe': 'income',
                'age_woe': 'age',
                'credit_score_woe': 'credit_score'
            }), self.model, only_total_score=True)
            
            return self.model, self.card
        except Exception as e:
            print(f"[ERROR] 模型训练失败: {str(e)}")
            raise

    def evaluate_model(self, y_true, y_pred):
        """模型评估"""
        # 计算KS值
        perf = sc.perf_eva(y_true, y_pred)
        
        # 将评分卡结果转换为字典格式，并添加y列
        score_dict = {
            'train': pd.DataFrame({
                'score': self.card['score'],
                'y': y_true
            }),
            'test': pd.DataFrame({
                'score': self.card['score'],
                'y': y_true
            })
        }
        
        # 绘制PSI曲线
        plt.figure(figsize=(8, 6))
        psi_result = sc.perf_psi(score_dict)
        plt.title('PSI Curve')
        plt.show()
        
        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        sc.perf_eva(y_true, y_pred)
        plt.title('ROC Curve')
        plt.show()
        
        return perf

    def predict(self, new_data):
        """对新数据进行预测和评分"""
        if self.model is None:
            raise ValueError("model untrained")
            
        # WOE转换
        woe_data = sc.woebin_ply(new_data, self.bins)
        
        # 计算评分
        scores = sc.scorecard_ply(woe_data, self.model)
        
        return scores

    def save_model(self, model_path):
        """保存模型"""
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump({'bins': self.bins, 'model': self.model}, f)

    def load_model(self, model_path):
        """加载模型"""
        import pickle
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
            self.bins = model_dict['bins']
            self.model = model_dict['model']
