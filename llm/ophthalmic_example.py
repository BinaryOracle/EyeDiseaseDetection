import json
from llm_manager import create_llm_manager

def example_ophthalmic_report():
    """示例：使用眼科报告模板生成报告"""
    
    # 创建LLM管理器
    manager = create_llm_manager()
    
    # 示例输入数据
    eye_examination_data = {
        "左眼报告": {
            "拍摄位置": "以黄斑为中心",
            "黄斑区": "樱桃红点",
            "视盘": "无明显异常",
            "血管": "微动脉瘤",
            "视网膜": "纤维增殖膜",
            "眼底疾病预测": "AMD"
        },
        "右眼报告": {
            "拍摄位置": "周边部",
            "黄斑区": "色素沉着",
            "视盘": "视盘边界模糊",
            "血管": "渗出",
            "视网膜": "广泛陈旧性激光斑",
            "眼底疾病预测": "RP"
        }
    }
    
    # 开始会话并使用眼科报告模板
    manager.start_session("ophthalmic_session")
    
    # 生成报告
    result = manager.chat_in_session(
        session_id="ophthalmic_session",
        input_variables={
            "eye_examination_data": json.dumps(eye_examination_data, ensure_ascii=False, indent=2)
        }
    )
    
    print("=== 眼科报告生成结果 ===")
    print("输入数据：")
    print(json.dumps(eye_examination_data, ensure_ascii=False, indent=2))
    print("\n生成的标准化报告：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 结束会话
    manager.end_session("ophthalmic_session")
    
    return result

if __name__ == "__main__":
    print("眼科报告生成示例")
    print("=" * 50)
    example_ophthalmic_report()
    print("\n示例运行完成！")