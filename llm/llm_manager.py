import os
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from llm_utils import LLMConversation

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMManager:
    """LLM 管理器，提供高级的对话管理和模板管理功能"""
    
    def __init__(self, config_file: str = None , template_name: str = "prompt_templates/ophthalmic_report"):
        """初始化 LLM 管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'config', 
            'llm_config.json'
        )
        
        # 加载配置
        self.config = self._load_config()
        
        # 创建 LLM 对话实例
        self.conversation = LLMConversation(
            base_url=self.config.get('base_url'),
            api_key=self.config.get('api_key'),
            model=self.config.get('model'),
            temperature=self.config.get('temperature'),
            max_tokens=self.config.get('max_tokens')
        )

        # 对话会话管理
        self.sessions: Dict[str, Dict[str, Any]] = {}

        # 默认模版key管理
        self.template_instance = self.get_nested(self.config, template_name)
        self.system_message= self.template_instance["system_message"]
        self.template = self.template_instance["template"]
        self.output_format = self.template_instance["output_format"]
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {e}")
            return {}

    def get_nested(self, config, key_path, sep="/"):
        keys = key_path.split(sep)
        value = config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return None
        return value

    def start_session(self, session_id: str) -> bool:
        """开始新的对话会话
        
        Args:
            session_id: 会话ID
            template_name: 使用的模板名称
            
        Returns:
            bool: 是否成功
        """
        if session_id in self.sessions:
            logger.warning(f"⚠️  会话 '{session_id}' 已存在")
            return False
        
        self.sessions[session_id] = {
            'started_at': datetime.now().isoformat(),
            'message_count': 0,
            'history': []
        }
        
        logger.info(f"✅ 会话 '{session_id}' 开始")
        return True
    
    def chat_in_session(self, 
                       session_id: str,
                       input_variables: Dict[str, Any] = None) -> Dict[str, Any]:

        if session_id not in self.sessions:
            logger.error(f"❌ 会话 '{session_id}' 不存在")
            return {"error": f"会话 '{session_id}' 不存在"}
        
        session = self.sessions[session_id]

        # 执行对话
        result = self.conversation.chat(
            prompt_template=self.template,
            input_variables=input_variables or {},
            output_format=self.output_format,
            system_message=self.system_message
        )
        
        # 更新会话信息
        session['message_count'] += 1
        session['history'].append({
            'timestamp': datetime.now().isoformat(),
            'input_variables': input_variables,
            'result': result
        })
        
        logger.info(f"✅ 会话 '{session_id}' 第{session['message_count']}次对话完成")
        return result
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """结束会话并返回会话摘要
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict: 会话摘要
        """
        if session_id not in self.sessions:
            logger.error(f"❌ 会话 '{session_id}' 不存在")
            return {"error": f"会话 '{session_id}' 不存在"}
        
        session = self.sessions.pop(session_id)
        
        summary = {
            'session_id': session_id,
            'started_at': session['started_at'],
            'ended_at': datetime.now().isoformat(),
            'message_count': session['message_count'],
            'history': session['history'],
            'duration_seconds': self._calculate_duration(session['started_at'])
        }
        
        logger.info(f"✅ 会话 '{session_id}' 结束，共进行 {session['message_count']} 次对话")
        return summary
    
    def _calculate_duration(self, start_time: str) -> float:
        """计算会话持续时间"""
        start = datetime.fromisoformat(start_time)
        end = datetime.now()
        return (end - start).total_seconds()
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            Optional[Dict]: 会话信息
        """
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> List[str]:
        """列出所有活跃会话"""
        return list(self.sessions.keys())


# 创建 LLM 管理器的工厂函数
def create_llm_manager(config_file: str = None) -> LLMManager:
    """创建并返回配置好的 LLMManager 实例"""
    return LLMManager(config_file)
