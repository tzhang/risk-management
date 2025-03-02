from scorecard_model import ScorecardModel
import pandas as pd
import numpy as np

def generate_sample_data(n_samples=1000):
    """生成示例数据用于演示"""
    np.random.seed(42)
    
    # 生成特征
    age = np.random.normal(35, 10, n_samples)
    income = np.random.normal(50000, 20000, n_samples)
    credit_score = np.random.normal(700, 50, n_samples)
    
    # 生成目标变量（违约概率与特征相关）
    prob = 1 / (1 + np.exp(-(age/100 - 0.4 + (income-50000)/100000 + (credit_score-700)/1000)))
    default = np.random.binomial(1, prob)
    
    # 创建数据框
    data = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'default': default
    })
    
    return data

def main():
    # 初始化模型
    model = ScorecardModel()
    
    # 生成示例数据
    data = generate_sample_data()
    
    # 数据预处理
    woe_data, woe_cols = model.preprocess_data(data, target='default')
    
    # 训练模型
    trained_model, scorecard = model.train_model(woe_data, target='default')
    
    # 模型评估
    y_true = data['default']
    y_pred = scorecard['score']
    performance = model.evaluate_model(y_true, y_pred)
    
    # 保存模型
    model.save_model('scorecard_model.pkl')
    
    # 预测新数据
    new_data = generate_sample_data(n_samples=100)  # 生成新的测试数据
    scores = model.predict(new_data)
    print("\n新数据的评分结果:")
    print(scores.head())

if __name__ == '__main__':
    main()
