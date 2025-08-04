import argparse
import datetime
import json
from typing import Dict, Optional
import pika
from config import device
from finetune import load_finetuned_model
from predict import predict

class Patient:
    """患者信息类，存储患者基本信息和筛查记录"""

    def __init__(self, patient_id: str, name: str, age: int, gender: str, phone: str, photos: list):
        self.patient_id = patient_id
        self.name = name
        self.age = age
        self.gender = gender
        self.phone = phone
        self.photos = photos
        self.screening_records = []  # 存储历次筛查记录

    def add_screening_record(self, record: Dict):
        """添加筛查记录"""
        self.screening_records.append(record)

    def to_dict(self) -> Dict:
        """转换为字典，用于数据存储"""
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "phone": self.phone,
            "screening_records": self.screening_records
        }

    @classmethod
    def from_json(cls, data: dict) -> 'Patient':
        """从 JSON 字符串构建 Patient 实例"""
        patient = cls(
            patient_id=data['patient_id'],
            name=data['name'],
            age=data['age'],
            gender=data['gender'],
            phone=data['phone'],
            photos=data.get('photos', [])
        )
        # 可选：恢复历史筛查记录
        patient.screening_records = data.get('screening_records', [])
        return patient

class EyeScreeningSystem:
    """眼底筛查系统，管理患者信息和筛查数据"""

    def __init__(self):
        self.patients = {}  # 以patient_id为键存储患者信息
        self.ai_diagnosis_results = {}  # 存储AI诊断结果

    def register_patient(self, patient: Patient) -> bool:
        """注册患者"""
        if patient.patient_id in self.patients:
            print(f"患者ID {patient.patient_id} 已存在")
            return False

        self.patients[patient.patient_id] = patient
        print(f"患者 {patient.name} 注册成功")
        return True

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """获取患者信息"""
        return self.patients.get(patient_id)

    def add_screening_data(self, patient_id: str, eye_image_path: str,
                           vision_data: Dict, notes: str = "") -> bool:
        """添加筛查数据"""
        patient = self.get_patient(patient_id)
        if not patient:
            print(f"患者ID {patient_id} 不存在")
            return False

        # 创建筛查记录
        screening_record = {
            "screening_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "eye_image_path": eye_image_path,
            "vision_data": vision_data,
            "notes": notes,
            "ai_diagnosis": None,
            "doctor_review": None,
            "report_generated": False
        }

        patient.add_screening_record(screening_record)
        print(f"患者 {patient.name} 的筛查数据添加成功")
        return True

    def ai_diagnosis(self, patient_id: str, record_index: int, diagnosis_result: Dict) -> bool:
        """模拟AI诊断"""
        patient = self.get_patient(patient_id)
        if not patient or record_index >= len(patient.screening_records):
            return False

        # 存储AI诊断结果
        patient.screening_records[record_index]["ai_diagnosis"] = diagnosis_result
        self.ai_diagnosis_results[f"{patient_id}_{record_index}"] = diagnosis_result
        print(f"AI诊断完成: {diagnosis_result['conclusion']}")
        return True

    def doctor_review(self, patient_id: str, record_index: int, review_result: Dict) -> bool:
        """医生审核"""
        patient = self.get_patient(patient_id)
        if not patient or record_index >= len(patient.screening_records):
            return False

        # 存储医生审核结果
        patient.screening_records[record_index]["doctor_review"] = review_result
        patient.screening_records[record_index]["report_generated"] = True
        print(f"医生审核完成: {review_result['final_conclusion']}")
        return True

    def generate_report(self, patient_id: str, record_index: int) -> Optional[Dict]:
        """生成筛查报告"""
        patient = self.get_patient(patient_id)
        if not patient or record_index >= len(patient.screening_records):
            return None

        record = patient.screening_records[record_index]
        if not record["report_generated"]:
            print("报告未生成，请先完成医生审核")
            return None

        # 构建报告
        report = {
            "report_id": f"REP_{patient_id}_{record_index}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "patient_info": {
                "id": patient.patient_id,
                "name": patient.name,
                "age": patient.age,
                "gender": patient.gender
            },
            "screening_time": record["screening_time"],
            "ai_diagnosis": record["ai_diagnosis"],
            "doctor_review": record["doctor_review"],
            "recommendation": record["doctor_review"].get("recommendation", "定期复查")
        }

        print(f"报告生成成功: {report['report_id']}")
        return report

    def save_data(self, file_path: str) -> bool:
        """保存数据到文件"""
        try:
            data = {
                "patients": {pid: patient.to_dict() for pid, patient in self.patients.items()},
                "ai_diagnosis_results": self.ai_diagnosis_results,
                "save_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"数据已保存到 {file_path}")
            return True
        except Exception as e:
            print(f"保存数据失败: {str(e)}")
            return False

class AIDoctor:
     """AI医生,负责AI疾病预测,AI报告生成等任务"""
     def __init__(self):
         # 参数初始化
         parser = argparse.ArgumentParser()
         parser.add_argument('--model', type=str, default="RETFound_mae")
         parser.add_argument('--input_size', default=224, type=int)
         parser.add_argument('--nb_classes', default=6, type=int)
         parser.add_argument('--global_pool', default='token')
         parser.add_argument('--drop_path', type=float, default=0.2)
         parser.add_argument('--save_path', type=str, default="output")
         self.args = parser.parse_args()

         # 加载模型
         self. model = load_finetuned_model(self.args, device)

     def ai_predict(self, img_path):
         """进行疾病预测"""
         return predict(img_path, self.model, self.args)

class MsgQueue:
    """协调多个子模块的交互过程"""
    def __init__(self):
        # 初始化RabbitMQ客户端
        self.rabbitmq_credentials = pika.PlainCredentials('guest', 'guest')
        self.rabbitmq_connection = pika.BlockingConnection(
            pika.ConnectionParameters('localhost', 5672, '/', self.rabbitmq_credentials)
        )
        self.rabbitmq_channel = self.rabbitmq_connection.channel()
        
        # 声明一个默认的交换机和队列用于消息传递
        self.default_exchange = 'eye_disease_detection_exchange'
        self.default_queue = 'eye_disease_detection_queue'
        
        # 声明交换机和队列（持久化）
        self.rabbitmq_channel.exchange_declare(exchange=self.default_exchange, exchange_type='direct', durable=True)
        self.rabbitmq_channel.queue_declare(queue=self.default_queue, durable=True)
        self.rabbitmq_channel.queue_bind(queue=self.default_queue, exchange=self.default_exchange, routing_key=self.default_queue)
        
    def send_message(self, message: str, routing_key: str = None):
        """发送消息到RabbitMQ队列"""
        if routing_key is None:
            routing_key = self.default_queue
            
        self.rabbitmq_channel.basic_publish(
            exchange=self.default_exchange,
            routing_key=routing_key,
            body=message,
            properties=pika.BasicProperties(delivery_mode=2)  # 消息持久化
        )
        print(f"消息已发送到队列: {message}")
        
    def close_connection(self):
        """关闭RabbitMQ连接"""
        if self.rabbitmq_connection and not self.rabbitmq_connection.is_closed:
            self.rabbitmq_connection.close()
            print("RabbitMQ连接已关闭")

    def get_one_message(self, auto_ack=False):
        """从RabbitMQ队列中获取一条消息（拉取方式）"""
        method_frame, header_frame, body = self.rabbitmq_channel.basic_get(
            queue=self.default_queue,
            auto_ack=auto_ack
        )
        if method_frame:
            try:
                message_data = json.loads(body.decode('utf-8'))
                print(f"拉取到JSON消息: {json.dumps(message_data, ensure_ascii=False, indent=2)}")
                return message_data
            except json.JSONDecodeError:
                print(f"拉取到消息: {body.decode('utf-8')}")

            if not auto_ack:
                self.rabbitmq_channel.basic_ack(delivery_tag=method_frame.delivery_tag)

            return None # JSON 解码失败
        else:
            print("队列中没有消息")
            return None

# 示例使用
if __name__ == "__main__":
    # 创建筛查系统
    system = EyeScreeningSystem()
    # AI 医生
    ai_doctor = AIDoctor()
    # 创建Manager实例
    manager = MsgQueue()

    # 注册患者
    msg = manager.get_one_message()
    patient = Patient.from_json(msg)
    system.register_patient(patient)

    # 添加筛查数据
    vision_data = {
        "left_eye": 0.8,
        "right_eye": 0.6,
        "intraocular_pressure": {"left": 18, "right": 19}
    }

    system.add_screening_data("P2025001", patient.photos[0], vision_data)

    # AI诊断
    result,confidence = ai_doctor.ai_predict(patient.photos[0])
    ai_result = {
        "conclusion": result,
        "confidence": confidence,
        "abnormal_areas": ["视网膜周边"]
    }

    system.ai_diagnosis("P2025001", 0, ai_result)

    # 医生审核
    doctor_review = {
        "final_conclusion": "高度近视",
        "doctor_name": "唐医生",
        "recommendation": "3个月后复查，避免剧烈运动"
    }
    system.doctor_review("P2025001", 0, doctor_review)

    # 生成报告
    report = system.generate_report("P2025001", 0)
    if report:
        print("\n生成的报告内容:")
        print(json.dumps(report, ensure_ascii=False, indent=2))

    # 保存数据
    system.save_data("diagnose_output/eye_screening_data.json")

    # 关闭RabbitMQ连接
    manager.close_connection()
    
    # 演示接收消息（注释掉以避免阻塞程序执行）
    # print("开始接收消息...")
    # manager.receive_messages()
