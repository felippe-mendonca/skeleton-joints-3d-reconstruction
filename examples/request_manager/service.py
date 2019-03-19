import time
from random import randint

from is_wire.core import Channel
from is_wire.rpc import ServiceProvider
from is_wire.rpc.log_interceptor import LogInterceptor

from is_msgs.common_pb2 import Pose, Position

mean_svc_time_ms = 100
var_src_time_ms = 20

min_svc_time_ms = mean_svc_time_ms - var_src_time_ms
max_svc_time_ms = mean_svc_time_ms + var_src_time_ms

channel = Channel('amqp://localhost:5672')
service_provider = ServiceProvider(channel)
service_provider.add_interceptor(LogInterceptor())


def service(pose, ctx):
    delay = randint(min_svc_time_ms, max_svc_time_ms) / 1000.0
    time.sleep(delay)
    return pose.position


service_provider.delegate(
    topic="GetPosition", function=service, request_type=Pose, reply_type=Position)

service_provider.run()
