import json
import pika

image_path = "/data/A/CAI_CHENGHUI_19460619_20201209_1910_IMAGEnetR4_Image_OD_1.2.392.200106.1651.4.2.200217210022131239.1607483470.153.tiff"

# 患者信息和图片路径
message = {
    "patient_id": "P2025001",
    "name": "扎西",
    "age": 45,
    "gender": "男",
    "phone": "13800138000",
    "photos": [image_path]
}

# 初始化RabbitMQ客户端
credentials = pika.PlainCredentials('guest', 'guest')
connection = pika.BlockingConnection(
    pika.ConnectionParameters('localhost', 5672, '/', credentials)
)
channel = connection.channel()

# 声明交换机和队列
exchange_name = 'eye_disease_detection_exchange'
queue_name = 'eye_disease_detection_queue'

channel.exchange_declare(exchange=exchange_name, exchange_type='direct', durable=True)
channel.queue_declare(queue=queue_name, durable=True)
channel.queue_bind(queue=queue_name, exchange=exchange_name, routing_key=queue_name)

# 发送消息
channel.basic_publish(
    exchange=exchange_name,
    routing_key=queue_name,
    body=json.dumps(message, ensure_ascii=False),
    properties=pika.BasicProperties(delivery_mode=2)  # 消息持久化
)

print(f"消息已发送到队列: {json.dumps(message, ensure_ascii=False, indent=2)}")

# 关闭连接
connection.close()