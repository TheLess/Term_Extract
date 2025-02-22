import pandas as pd
import numpy as np
from pathlib import Path
import json

def generate_game_terms():
    """生成游戏相关的示例术语"""
    prefixes = ['攻击力', '防御力', '魔法', '生命值', '耐力', '敏捷', '智力']
    suffixes = ['提升', '降低', '恢复', '减少', '加成', '弱化']
    numbers = ['', '+5', '+10', '-5', '-10']
    
    terms = []
    for prefix in prefixes:
        # 基础术语
        terms.append(prefix)
        # 带数值的术语
        for num in numbers:
            if num:
                terms.append(f"{prefix}{num}")
        # 带效果的术语
        for suffix in suffixes:
            terms.append(f"{prefix}{suffix}")
            # 带数值和效果的术语
            for num in numbers:
                if num:
                    terms.append(f"{prefix}{suffix}{num}")
    
    return terms

def translate_to_english(term):
    """将中文术语翻译为英文（示例翻译）"""
    translations = {
        '攻击力': 'Attack',
        '防御力': 'Defense',
        '魔法': 'Magic',
        '生命值': 'HP',
        '耐力': 'Stamina',
        '敏捷': 'Agility',
        '智力': 'Intelligence',
        '提升': 'Up',
        '降低': 'Down',
        '恢复': 'Recovery',
        '减少': 'Decrease',
        '加成': 'Boost',
        '弱化': 'Weaken'
    }
    
    # 处理数值
    for num in ['+5', '+10', '-5', '-10']:
        if num in term:
            term = term.replace(num, ' ' + num)
    
    # 翻译各部分
    for cn, en in translations.items():
        term = term.replace(cn, en)
    
    return term

def translate_to_japanese(term):
    """将中文术语翻译为日文（示例翻译）"""
    translations = {
        '攻击力': '攻撃力',
        '防御力': '防御力',
        '魔法': '魔法',
        '生命值': 'HP',
        '耐力': 'スタミナ',
        '敏捷': '素早さ',
        '智力': '知力',
        '提升': 'アップ',
        '降低': 'ダウン',
        '恢复': '回復',
        '减少': '減少',
        '加成': 'ブースト',
        '弱化': '弱体化'
    }
    
    # 处理数值
    for num in ['+5', '+10', '-5', '-10']:
        if num in term:
            term = term.replace(num, num)
    
    # 翻译各部分
    for cn, ja in translations.items():
        term = term.replace(cn, ja)
    
    return term

def translate_to_korean(term):
    """将中文术语翻译为韩文（示例翻译）"""
    translations = {
        '攻击力': '공격력',
        '防御力': '방어력',
        '魔法': '마법',
        '生命值': 'HP',
        '耐力': '지구력',
        '敏捷': '민첩성',
        '智力': '지력',
        '提升': '증가',
        '降低': '감소',
        '恢复': '회복',
        '减少': '감소',
        '加成': '강화',
        '弱化': '약화'
    }
    
    # 处理数值
    for num in ['+5', '+10', '-5', '-10']:
        if num in term:
            term = term.replace(num, ' ' + num)
    
    # 翻译各部分
    for cn, ko in translations.items():
        term = term.replace(cn, ko)
    
    return term

def generate_sample_data(output_dir: Path):
    """生成示例数据文件"""
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成术语
    terms = generate_game_terms()
    
    # 创建翻译数据
    data = {
        '简体中文': terms,
        'English': [translate_to_english(term) for term in terms],
        '日本語': [translate_to_japanese(term) for term in terms],
        '한국어': [translate_to_korean(term) for term in terms]
    }
    
    # 保存为Excel文件
    df = pd.DataFrame(data)
    excel_file = output_dir / 'sample_translations.xlsx'
    df.to_excel(excel_file, index=False)
    print(f"已生成示例翻译文件: {excel_file}")
    
    # 生成术语库
    term_db = {term: np.random.randint(1, 10) for term in terms}
    term_db_file = output_dir / 'sample_term_db.json'
    with open(term_db_file, 'w', encoding='utf-8') as f:
        json.dump(term_db, f, ensure_ascii=False, indent=2)
    print(f"已生成示例术语库: {term_db_file}")

if __name__ == '__main__':
    # 生成示例数据
    output_dir = Path(__file__).parent.parent / 'data' / 'sample'
    generate_sample_data(output_dir)
    print("示例数据生成完成！")
