import pika
import json
import logging
import os
from typing import List, Dict, Any, Optional, Callable

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置文件路径
CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'rabbitmq_config.json')

# 队列和交换机常量定义
CHECK_QUEUE = "check_queue"
REPORT_QUEUE = "report_queue"
RESULT_CHECK_QUEUE = "result_check_queue"
RESULT_REPORT_QUEUE = "result_report_queue"

CHECK_EXCHANGE = "check_exchange"
REPORT_EXCHANGE = "report_exchange"
RESULT_CHECK_EXCHANGE = "result_check_exchange"
RESULT_REPORT_EXCHANGE = "result_report_exchange"

CHECK_ROUTING_KEY = "check_routing_key"
REPORT_ROUTING_KEY = "report_routing_key"
RESULT_CHECK_ROUTING_KEY = "result_check_routing_key"
RESULT_REPORT_ROUTING_KEY = "result_report_routing_key"

# 所有队列和交换机的配置映射
QUEUE_EXCHANGE_MAP = {
    CHECK_QUEUE: {
        'exchange': CHECK_EXCHANGE,
        'routing_key': CHECK_ROUTING_KEY
    },
    REPORT_QUEUE: {
        'exchange': REPORT_EXCHANGE,
        'routing_key': REPORT_ROUTING_KEY
    },
    RESULT_CHECK_QUEUE: {
        'exchange': RESULT_CHECK_EXCHANGE,
        'routing_key': RESULT_CHECK_ROUTING_KEY
    },
    RESULT_REPORT_QUEUE: {
        'exchange': RESULT_REPORT_EXCHANGE,
        'routing_key': RESULT_REPORT_ROUTING_KEY
    }
}


def load_config() -> Dict[str, Any]:
    """加载 RabbitMQ 配置文件"""
    try:
        with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info("✅ RabbitMQ 配置文件加载成功")
        return config.get('rabbitmq', {})
    except FileNotFoundError:
        logger.error(f"❌ RabbitMQ 配置文件未找到: {CONFIG_FILE_PATH}")
        raise
    except json.JSONDecodeError:
        logger.error("❌ RabbitMQ 配置文件格式错误")
        raise
    except Exception as e:
        logger.error(f"❌ 加载 RabbitMQ 配置失败: {e}")
        raise


class RabbitMQUtils:
    """RabbitMQ 工具类，提供基础的连接、发送和接收消息功能"""

    def __init__(self, host: str = None, port: int = None,
                 username: str = None, password: str = None,
                 auto_initialize: bool = True):
        config = load_config()

        self.host = host or config.get('host', '47.108.62.179')
        self.port = port or config.get('port', 5672)
        self.username = username or config.get('username', 'guest')
        self.password = password or config.get('password', 'guest')
        self.connection = None
        self.channel = None

        self.queue_config = config.get('queues', {})

        if auto_initialize:
            self._auto_initialize_queues_exchanges()

    def _auto_initialize_queues_exchanges(self):
        """自动初始化队列和交换机"""
        if not self.connect():
            logger.warning("⚠️  连接失败，无法自动初始化队列和交换机")
            return

        try:
            # 先创建所有交换机
            for exchange in [CHECK_EXCHANGE, REPORT_EXCHANGE, RESULT_CHECK_EXCHANGE, RESULT_REPORT_EXCHANGE]:
                self.channel.exchange_declare(exchange=exchange, exchange_type='direct', durable=True)
                logger.info(f"✅ 初始化交换机: {exchange}")

            # 再创建所有队列并绑定到交换机
            for queue_name, config in QUEUE_EXCHANGE_MAP.items():
                self.channel.queue_declare(queue=queue_name, durable=True)
                self.channel.queue_bind(exchange=config['exchange'], queue=queue_name,
                                        routing_key=config['routing_key'])
                logger.info(f"✅ 初始化并绑定队列: {queue_name} -> {config['exchange']}")

            logger.info("✅ 队列和交换机初始化完成")
        except Exception as e:
            logger.error(f"❌ 初始化队列和交换机失败: {e}")

    def connect(self) -> bool:
        """建立 RabbitMQ 连接"""
        try:
            credentials = pika.PlainCredentials(self.username, self.password)
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=self.host,
                    port=self.port,
                    credentials=credentials,
                    heartbeat=600,
                    blocked_connection_timeout=300
                )
            )
            self.channel = self.connection.channel()
            logger.info(f"✅ 成功连接到 RabbitMQ: {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"❌ 连接 RabbitMQ 失败: {e}")
            return False

    def disconnect(self):
        """关闭 RabbitMQ 连接"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("✅ RabbitMQ 连接已关闭")

    def send_single_message(self, exchange: str, routing_key: str, message: Dict[str, Any],
                            queue: Optional[str] = None) -> bool:
        """发送单条消息"""
        if not self.channel or self.connection.is_closed:
            if not self.connect():
                return False

        try:
            if queue:
                # 确保队列存在
                self.channel.queue_declare(queue=queue, durable=True)
                self.channel.queue_bind(exchange=exchange, queue=queue, routing_key=routing_key)

            self.channel.basic_publish(
                exchange=exchange,
                routing_key=routing_key,
                body=json.dumps(message, ensure_ascii=False),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    content_type='application/json'
                )
            )
            logger.info(f"✅ 消息发送成功 - 交换机: {exchange}, 路由键: {routing_key}")
            return True
        except Exception as e:
            logger.error(f"❌ 消息发送失败: {e}")
            return False

    def send_batch_messages(self, exchange: str, routing_key: str, messages: List[Dict[str, Any]],
                            queue: Optional[str] = None) -> bool:
        """批量发送消息"""
        if not messages:
            logger.warning("⚠️  消息列表为空，跳过发送")
            return True

        if not self.channel or self.connection.is_closed:
            if not self.connect():
                return False

        try:
            if queue:
                self.channel.queue_declare(queue=queue, durable=True)
                self.channel.queue_bind(exchange=exchange, queue=queue, routing_key=routing_key)

            success_count = 0
            for message in messages:
                self.channel.basic_publish(
                    exchange=exchange,
                    routing_key=routing_key,
                    body=json.dumps(message, ensure_ascii=False),
                    properties=pika.BasicProperties(
                        delivery_mode=2,
                        content_type='application/json'
                    )
                )
                success_count += 1

            logger.info(f"✅ 批量发送成功 - 共 {success_count}/{len(messages)} 条消息")
            return success_count == len(messages)
        except Exception as e:
            logger.error(f"❌ 批量发送失败: {e}")
            return False

    def get_single_message(self, queue: str, auto_ack: bool = True) -> Optional[Dict[str, Any]]:
        """非阻塞获取单条消息"""
        if not self.channel or self.connection.is_closed:
            if not self.connect():
                return None
        try:
            method_frame, _, body = self.channel.basic_get(queue=queue, auto_ack=auto_ack)
            if method_frame:
                message = json.loads(body.decode('utf-8'))
                logger.info(f"📨 获取到消息 - 队列: {queue}")
                return message
            else:
                logger.info(f"ℹ️  队列 {queue} 中没有消息")
                return None
        except Exception as e:
            logger.error(f"❌ 获取消息失败: {e}")
            return None

    def get_batch_messages(self, queue: str, batch_size: int = 10, auto_ack: bool = True) -> List[Dict[str, Any]]:
        """非阻塞批量获取消息"""
        messages = []
        if not self.channel or self.connection.is_closed:
            if not self.connect():
                return messages

        try:
            for _ in range(batch_size):
                method_frame, _, body = self.channel.basic_get(queue=queue, auto_ack=auto_ack)
                if method_frame:
                    message = json.loads(body.decode('utf-8'))
                    messages.append(message)
                else:
                    break
            logger.info(f"📨 批量获取消息 - 队列: {queue}, 获取数量: {len(messages)}")
            return messages
        except Exception as e:
            logger.error(f"❌ 批量获取消息失败: {e}")
            return messages

    def get_message_count(self, queue: str) -> int:
        """获取队列消息数量"""
        if not self.channel or self.connection.is_closed:
            if not self.connect():
                return -1
        try:
            queue_declare_ok = self.channel.queue_declare(queue=queue, passive=True)
            count = queue_declare_ok.method.message_count
            logger.info(f"📊 队列 {queue} 中有 {count} 条消息")
            return count
        except Exception as e:
            logger.error(f"❌ 获取消息数量失败: {e}")
            return -1

    def get_queue_config(self, queue_name: str) -> Optional[Dict[str, str]]:
        """获取队列配置"""
        return self.queue_config.get(queue_name)


def create_rabbitmq_utils() -> RabbitMQUtils:
    return RabbitMQUtils()


if __name__ == "__main__":
    mq_utils = create_rabbitmq_utils()

    if mq_utils.connect():
        check_message = {
            "osAnteriorImg": "/admin/sys-file/aipluseyes/sample_os_anterior.png",
            "odAnteriorImg": "/admin/sys-file/aipluseyes/sample_od_anterior.png",
            "osFundusImg": "/admin/sys-file/aipluseyes/sample_os_fundus.png",
            "odFundusImg": "/admin/sys-file/aipluseyes/sample_od_fundus.png",
            "id": "test_123456789"
        }

        queue_config = mq_utils.get_queue_config('check_queue')
        if queue_config:
            success = mq_utils.send_single_message(
                exchange=queue_config['exchange'],
                routing_key=queue_config['routing_key'],
                message=check_message,
                queue='check_queue'
            )
            print("✅ 测试消息发送成功" if success else "❌ 测试消息发送失败")

        count = mq_utils.get_message_count('check_queue')
        print(f"当前检查队列消息数量: {count}")

        msg = mq_utils.get_single_message(queue='check_queue', auto_ack=True)
        print(f"拉取到的消息: {msg}")

        mq_utils.disconnect()
