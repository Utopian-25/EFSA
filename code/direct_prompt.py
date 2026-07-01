import os
import json
from typing import List, Tuple, Dict, Any
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer, AutoModel

class FinancialEventAnalyzer:
    """金融事件细粒度情感分析器"""
    
    def __init__(self, model_name: str):
        """初始化模型和tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
        self.model = self.model.eval()
        
        # 定义事件层级结构
        self.finance_event_hierarchy = {
            '财务': ['利润公布', '利润预告', '其他财务动态'],
            '股东': ['股东增减持', '股东质押', '其他股东事件'],
            '股票': ['股价变动', '股票状态', '限售股解禁', '股票回购', '股权激励&员工持股计划', 
                    '限制股解禁', '股票分红', '其他股票事件'],
            '管理': ['董监高动态', '员工动态'],
            '合规信用': ['监管问询', '公司涉诉', '立案调查', '行政处罚', '澄清公告', 
                       '法律事务', '评级变动', '其他违规事件'],
            '经营问题': ['项目中标', '其他经营事件', '开展合作', '新公司成立', '销量、份额变动', 
                       '知识产权', '技术质控、资质变动', '政府补贴', '机构调研', '产能产量变动', 
                       '项目动态', '产品动态'],
            '融资投资': ['资金流动', '投资事件', '融资融券', '公司上市', '并购重组', 
                       '股票增发', '其他融资事件']
        }
        
        # 定义情感极性
        self.sentiments = ['积极', '中性', '消极']
    
    def extract_companies(self, news_text: str) -> List[str]:
        """Step 1: 提取新闻中涉及的所有公司名称"""
        prompt = f"""
        假设你是一个金融领域的实体识别模型，请从以下金融新闻中识别出所有涉及的公司名称。
        只返回公司名称列表，用逗号分隔，不要回答多余的话。
        
        金融新闻：{news_text}
        """
        response, _ = self.model.chat(self.tokenizer, prompt, history=None)
        # 解析公司名称列表
        companies = [name.strip() for name in response.replace('[', '').replace(']', '').split(',') if name.strip()]
        return companies if companies else [self._extract_single_company(news_text)]
    
    def _extract_single_company(self, news_text: str) -> str:
        """备选方案：提取单个公司名称"""
        prompt = f"""
        从以下金融新闻中识别出主要的公司名称，只回答公司名称，不要多余的话。
        金融新闻：{news_text}
        """
        response, _ = self.model.chat(self.tokenizer, prompt, history=None)
        return response.strip()
    
    def extract_coarse_events(self, news_text: str, company: str) -> List[str]:
        """Step 2: 识别某公司涉及的所有一级事件"""
        coarse_events = list(self.finance_event_hierarchy.keys())
        prompt = f"""
        假设你是一个金融领域的事件分类模型，请分析以下新闻中公司"{company}"涉及的所有一级事件类型。
        必须从以下列表中选择，可以选多个：{coarse_events}
        只返回事件类型列表，用逗号分隔，不要回答多余的话。
        
        金融新闻：{news_text}
        """
        response, _ = self.model.chat(self.tokenizer, prompt, history=None)
        # 解析事件列表
        events = [event.strip() for event in response.replace('[', '').replace(']', '').split(',') 
                 if event.strip() in coarse_events]
        return events if events else [self._extract_single_coarse_event(news_text, company)]
    
    def _extract_single_coarse_event(self, news_text: str, company: str) -> str:
        """备选方案：提取单个一级事件"""
        coarse_events = list(self.finance_event_hierarchy.keys())
        prompt = f"""
        从以下新闻中判断公司"{company}"涉及的一级事件类型，从{coarse_events}中选择一个。
        只回答事件类型，不要多余的话。
        金融新闻：{news_text}
        """
        response, _ = self.model.chat(self.tokenizer, prompt, history=None)
        # 确保返回的事件在预定义列表中
        for event in coarse_events:
            if event in response:
                return event
        return coarse_events[0]  # 默认返回第一个
    
    def extract_fine_events(self, news_text: str, company: str, coarse_event: str) -> List[str]:
        """Step 3: 识别某公司-一级事件分支下的所有二级事件"""
        fine_events = self.finance_event_hierarchy.get(coarse_event, [])
        if not fine_events:
            return ['其他事件']
            
        prompt = f"""
        假设你是一个金融领域的细粒度事件分类模型，请分析以下新闻中公司"{company}"发生的一级事件"{coarse_event}"下涉及的二级事件类型。
        必须从以下列表中选择，可以选多个：{fine_events}
        只返回事件类型列表，用逗号分隔，不要回答多余的话。
        
        金融新闻：{news_text}
        """
        response, _ = self.model.chat(self.tokenizer, prompt, history=None)
        # 解析事件列表
        events = [event.strip() for event in response.replace('[', '').replace(']', '').split(',') 
                 if event.strip() in fine_events]
        return events if events else [self._extract_single_fine_event(news_text, company, coarse_event)]
    
    def _extract_single_fine_event(self, news_text: str, company: str, coarse_event: str) -> str:
        """备选方案：提取单个二级事件"""
        fine_events = self.finance_event_hierarchy.get(coarse_event, [])
        if not fine_events:
            return '其他事件'
            
        prompt = f"""
        从以下新闻中判断公司"{company}"发生的一级事件"{coarse_event}"下属于哪个二级事件，从{fine_events}中选择一个。
        只回答事件类型，不要多余的话。
        金融新闻：{news_text}
        """
        response, _ = self.model.chat(self.tokenizer, prompt, history=None)
        # 确保返回的事件在预定义列表中
        for event in fine_events:
            if event in response:
                return event
        return fine_events[0]  # 默认返回第一个
    
    def extract_sentiment(self, news_text: str, company: str, coarse_event: str, fine_event: str) -> str:
        """Step 4: 判断具体事件的情感极性"""
        prompt = f"""
        假设你是一个金融领域的细粒度情感分析模型，请判断以下新闻中对公司"{company}"发生的"{coarse_event}"-"{fine_event}"事件的情感倾向。
        从以下情感极性列表中选择：{self.sentiments}
        只返回情感极性，不要回答多余的话。
        
        金融新闻：{news_text}
        """
        response, _ = self.model.chat(self.tokenizer, prompt, history=None)
        # 确保返回的情感极性在预定义列表中
        for sentiment in self.sentiments:
            if sentiment in response:
                return sentiment
        return '中性'  # 默认返回中性
    
    def analyze_news(self, news_text: str) -> List[Tuple[str, str, str, str]]:
        """完整的树形展开分析流程"""
        results = []
        
        # Step 1: 提取所有公司
        companies = self.extract_companies(news_text)
        print(f"识别到的公司: {companies}")
        
        # 对每家公司分别处理
        for company in companies:
            # Step 2: 提取该公司的所有一级事件
            coarse_events = self.extract_coarse_events(news_text, company)
            print(f"公司 {company} 的一级事件: {coarse_events}")
            
            # 对每个一级事件分支处理
            for coarse_event in coarse_events:
                # Step 3: 提取该分支下的所有二级事件
                fine_events = self.extract_fine_events(news_text, company, coarse_event)
                print(f"公司 {company}-{coarse_event} 的二级事件: {fine_events}")
                
                # 对每个二级事件分支处理
                for fine_event in fine_events:
                    # Step 4: 判断情感极性
                    sentiment = self.extract_sentiment(news_text, company, coarse_event, fine_event)
                    results.append((company, coarse_event, fine_event, sentiment))
                    print(f"生成四元组: {company}, {coarse_event}, {fine_event}, {sentiment}")
        
        # 去重
        return list(set(results))

def main():
    """主函数"""
    # 初始化分析器
    analyzer = FinancialEventAnalyzer("model-name")  # 替换为实际的模型名称
    
    # 示例新闻
    news_text = """浦东建设公告,近日公司子公司上海市浦东新区建设(集团)有限公司、上海浦东路桥(集团)有限公司中标多项重大工程项目,中标金额总计15.66亿元。"""
    
    # 执行分析
    results = analyzer.analyze_news(news_text)
    
    # 输出结果
    print("\n最终分析结果:")
    for company, coarse, fine, sentiment in results:
        print(f"('{company}', '{coarse}', '{fine}', '{sentiment}')")
    
    # 如果需要补充行业信息，可以添加公司-行业映射
    # 最终得到五元组 (公司, 行业, 一级事件, 二级事件, 情感)

if __name__ == "__main__":
    main()
