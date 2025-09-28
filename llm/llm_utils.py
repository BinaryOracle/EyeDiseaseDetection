import os
import json
import logging
from typing import Dict, Any, List, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain_openai import ChatOpenAI

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置文件路径
CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'llm_config.json')


def load_llm_config() -> Dict[str, Any]:
    """加载 LLM 配置文件"""
    try:
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info("✅ LLM 配置文件加载成功")
        return config
    except FileNotFoundError:
        logger.error(f"❌ LLM 配置文件未找到: {CONFIG_FILE_PATH}")
        # 返回默认配置
        return {
            "provider": "openai",
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000
        }
    except json.JSONDecodeError:
        logger.error("❌ LLM 配置文件格式错误")
        raise
    except Exception as e:
        logger.error(f"❌ 加载 LLM 配置失败: {e}")
        raise


class StructuredOutputParser(BaseOutputParser):
    """结构化输出解析器"""
    
    def __init__(self, output_format: Dict[str,Any]):
        super().__init__()
        object.__setattr__(self, "output_format", output_format)
    
    def parse(self, text: str) -> Dict[str, Any]:
        """解析 LLM 输出为结构化数据"""
        try:
            # 尝试解析 JSON
            if text.strip().startswith('{') and text.strip().endswith('}'):
                return json.loads(text)
            
            # 如果无法解析为 JSON，返回原始文本
            return {"raw_output": text}
        except json.JSONDecodeError:
            logger.warning("⚠️  LLM 输出无法解析为 JSON，返回原始文本")
            return {"raw_output": text}
    
    def get_format_instructions(self) -> str:
        """获取格式说明"""
        # 将JSON格式中的花括号转义，避免被LangChain识别为输入变量
        json_example = json.dumps(self.output_format, ensure_ascii=False, indent=2)
        escaped_json = json_example.replace("{", "{{").replace("}", "}}")
        
        return f"""请严格按照以下 JSON 格式输出结果：
{escaped_json}

确保输出是有效的 JSON 格式，不要包含任何额外的文本或解释。"""


class LLMConversation:
    """LLM 对话工具类"""
    
    def __init__(self, 
                 base_url: str = None,
                 api_key: str = None,
                 model: str = None,
                 temperature: float = None,
                 max_tokens: int = None):
        config = load_llm_config()
        
        self.base_url = base_url or config.get('base_url')
        self.api_key = api_key or config.get('api_key') or os.getenv('OPENAI_API_KEY')
        self.model = model or config.get('model')
        self.temperature = temperature or config.get('temperature', 0.7)
        self.max_tokens = max_tokens or config.get('max_tokens', 1000)
        
        # 初始化 LLM
        self.llm = self._initialize_llm()
        
        # 存储对话历史
        self.conversation_history: List[Dict[str, str]] = []
    
    def _initialize_llm(self):
        """初始化 LLM 实例"""
        if not self.api_key:
            raise ValueError("❌ API 密钥未提供")
        
        return ChatOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
    
    def create_prompt_template(self, 
                              template: str, 
                              output_format: Optional[Dict[str, Any]] = None,
                              input_variables: List[str] = None) -> PromptTemplate:
        """创建提示模板
        
        Args:
            template: 提示模板内容
            output_format: 输出格式定义
            input_variables: 输入变量列表
            
        Returns:
            PromptTemplate: 配置好的提示模板
        """
        if output_format:
            parser = StructuredOutputParser(output_format)
            format_instructions = parser.get_format_instructions()
            
            # 使用双花括号转义JSON格式中的花括号，避免被识别为输入变量
            # 只转义模板内容，不转义格式说明（因为格式说明中的花括号是示例，不是输入变量）
            escaped_template = template.replace("{", "{{").replace("}", "}}")
            template_with_format = escaped_template + "\n\n" + format_instructions
            
            # 显式指定输入变量，避免LangChain自动扫描模板
            return PromptTemplate(
                template=template_with_format,
                input_variables=input_variables or ["eye_examination_data"],
                output_parser=parser
            )
        else:
            # 对于没有输出格式的模板，也需要转义花括号
            escaped_template = template.replace("{", "{{").replace("}", "}}")
            return PromptTemplate(
                template=escaped_template,
                input_variables=input_variables or []
            )
    
    def chat(self, 
            prompt_template: str,
            input_variables: Dict[str, Any] = None,
            output_format: Optional[Dict[str, Any]] = None,
            system_message: str = None) -> Dict[str, Any]:
        """与 LLM 进行对话
        
        Args:
            prompt_template: 提示模板
            input_variables: 输入变量
            output_format: 输出格式定义
            system_message: 系统消息
            
        Returns:
            Dict: 对话结果
        """
        try:
            # 构建完整的提示
            full_prompt = self._build_full_prompt(prompt_template, input_variables, system_message)
            
            # 创建提示模板
            template_variables = list(input_variables.keys()) if input_variables else []
            prompt_template_obj = self.create_prompt_template(
                template=full_prompt,
                output_format=output_format,
                input_variables=template_variables
            )
            
            # 创建 LLM 链
            chain = LLMChain(llm=self.llm, prompt=prompt_template_obj)
            
            # 执行对话
            if input_variables:
                result = chain.run(**input_variables)
            else:
                result = chain.run({})
            
            # 解析输出
            if output_format and hasattr(prompt_template_obj.output_parser, 'parse'):
                parsed_result = prompt_template_obj.output_parser.parse(result)
            else:
                parsed_result = {"response": result}
            
            # 记录对话历史
            self.conversation_history.append({
                "prompt": prompt_template,
                "input_variables": input_variables,
                "response": parsed_result
            })
            
            logger.info("✅ LLM 对话完成")
            return parsed_result
            
        except Exception as e:
            logger.error(f"❌ LLM 对话失败: {e}")
            return {"error": str(e)}
    
    def _build_full_prompt(self, 
                          prompt_template: str,
                          input_variables: Dict[str, Any] = None,
                          system_message: str = None) -> str:
        """构建完整的提示"""
        full_prompt = ""
        
        # 添加系统消息
        if system_message:
            full_prompt += f"系统指令: {system_message}\n\n"
        
        # 添加对话历史（如果有）
        if self.conversation_history:
            full_prompt += "对话历史:\n"
            for i, history in enumerate(self.conversation_history[-3:], 1):  # 只保留最近3轮
                full_prompt += f"第{i}轮: {history.get('prompt', '')} -> {history.get('response', {})}\n"
            full_prompt += "\n"
        
        # 添加当前提示
        full_prompt += prompt_template
        
        return full_prompt
    
    def clear_conversation_history(self):
        """清空对话历史"""
        self.conversation_history.clear()
        logger.info("✅ 对话历史已清空")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self.conversation_history.copy()

# 创建 LLM 对话实例的工厂函数
def create_llm_conversation(**kwargs) -> LLMConversation:
    """创建并返回配置好的 LLMConversation 实例"""
    return LLMConversation(**kwargs)