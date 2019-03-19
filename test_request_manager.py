from is_wire.core import Channel, Logger
from src.utils.is_wire import RequestManager

from is_msgs.common_pb2 import Pose, Position

channel = Channel('amqp://localhost:5672')
request_manager = RequestManager(channel, min_requests=-1, max_requests=30, log_level=Logger.INFO)

poses = [(Pose(position=Position(x=i, y=2 * i)), i) for i in range(100)]

while True:

    while request_manager.can_request() and len(poses) > 0:
        pose, pose_id = poses.pop()
        request_manager.request( content=pose, topic="GetPosition", timeout_ms=1000, metadata=pose_id)
        print(len(request_manager._requests))

    received_msgs = request_manager.consume_ready(timeout=1.0)

    for msg, metadata in received_msgs:
        position = msg.unpack(Position)
        print(metadata, position)
    else:
        pass

    if request_manager.all_received() and len(poses) == 0:
        break