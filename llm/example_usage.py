#!/usr/bin/env python3
"""
LLM 对话系统使用示例

这个示例展示了如何使用 LLM 工具类进行各种类型的对话。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.llm_manager import create_llm_manager
from llm.llm_utils import PromptTemplates


def setup_environment():
    """设置环境变量"""
    # 设置 OpenAI API 密钥（请替换为您的实际密钥）
    os.environ['OPENAI_API_KEY'] = 'your_openai_api_key_here'
    
    print("✅ 环境变量设置完成")


def example_simple_chat():
    """示例1：简单对话"""
    print("\n" + "="*50)
    print("示例1：简单对话")
    print("="*50)
    
    manager = create_llm_manager()
    
    # 开始一个简单会话
    session_id = "simple_chat_001"
    manager.start_session(session_id)
    
    # 进行对话
    result = manager.chat_in_session(
        session_id=session_id,
        custom_prompt="请用中文介绍人工智能在医疗领域的应用"
    )
    
    print("对话结果:")
    print(result.get('response', '无响应'))
    
    # 结束会话
    summary = manager.end_session(session_id)
    print(f"会话摘要: {summary}")


def example_medical_diagnosis():
    """示例2：医学诊断对话"""
    print("\n" + "="*50)
    print("示例2：医学诊断对话")
    print("="*50)
    
    manager = create_llm_manager()
    
    # 开始医学诊断会话
    session_id = "medical_diagnosis_001"
    manager.start_session(session_id, "medical_diagnosis")
    
    # 进行诊断对话
    result = manager.chat_in_session(
        session_id=session_id,
        input_variables={
            "analysis_results": "眼底图像显示视网膜有轻微出血，黄斑区有水肿",
            "symptoms": "视力模糊，眼前有黑影，眼干",
            "age": 58,
            "gender": "女"
        }
    )
    
    print("诊断结果:")
    for key, value in result.items():
        print(f"{key}: {value}")
    
    # 结束会话
    summary = manager.end_session(session_id)
    print(f"会话摘要: {summary}")


def example_custom_template():
    """示例3：使用自定义模板"""
    print("\n" + "="*50)
    print("示例3：使用自定义模板")
    print("="*50)
    
    manager = create_llm_manager()
    
    # 添加自定义模板
    manager.add_custom_template(
        template_name="eye_screening_report",
        template="""请根据眼部筛查数据生成报告：

筛查数据：
{screening_data}

患者信息：
- 年龄：{age}岁
- 性别：{gender}
- 筛查时间：{screening_time}

报告要求：
1. 筛查结果概述
2. 风险评估
3. 建议的随访计划
4. 健康建议""",
        system_message="你是一个专业的眼科筛查报告生成专家",
        output_format={
            "overview": "筛查结果概述",
            "risk_assessment": "风险评估",
            "follow_up_plan": "随访计划",
            "health_advice": "健康建议"
        },
        description="眼部筛查报告生成模板"
    )
    
    # 开始自定义模板会话
    session_id = "custom_template_001"
    manager.start_session(session_id, "eye_screening_report")
    
    # 使用自定义模板进行对话
    result = manager.chat_in_session(
        session_id=session_id,
        input_variables={
            "screening_data": "视力检查：左眼0.8，右眼0.6；眼压：左眼15mmHg，右眼18mmHg；眼底：视网膜正常",
            "age": 45,
            "gender": "男",
            "screening_time": "2024-01-15"
        }
    )
    
    print("筛查报告:")
    for key, value in result.items():
        print(f"{key}: {value}")
    
    # 结束会话
    summary = manager.end_session(session_id)
    print(f"会话摘要: {summary}")


def example_multiple_sessions():
    """示例4：多会话管理"""
    print("\n" + "="*50)
    print("示例4：多会话管理")
    print("="*50)
    
    manager = create_llm_manager()
    
    # 创建多个会话
    sessions = [
        ("session_medical", "medical_diagnosis"),
        ("session_report", "report_generation"),
        ("session_data", "data_analysis")
    ]
    
    for session_id, template_name in sessions:
        manager.start_session(session_id, template_name)
        print(f"✅ 会话 '{session_id}' 已开始")
    
    # 列出所有活跃会话
    active_sessions = manager.list_sessions()
    print(f"活跃会话: {active_sessions}")
    
    # 列出所有可用模板
    templates = manager.list_templates()
    print("可用模板类型:")
    for template_type, template_list in templates.items():
        print(f"  {template_type}: {template_list}")
    
    # 结束所有会话
    for session_id, _ in sessions:
        summary = manager.end_session(session_id)
        print(f"会话 '{session_id}' 已结束: {summary}")


def main():
    """主函数"""
    print("LLM 对话系统使用示例")
    print("请确保已设置正确的 OpenAI API 密钥")
    
    # 设置环境
    setup_environment()
    
    try:
        # 运行各个示例
        example_simple_chat()
        example_medical_diagnosis()
        example_custom_template()
        example_multiple_sessions()
        
        print("\n" + "="*50)
        print("所有示例运行完成！")
        print("="*50)
        
    except Exception as e:
        print(f"❌ 示例运行失败: {e}")
        print("请检查 API 密钥配置和网络连接")


if __name__ == "__main__":
    main()