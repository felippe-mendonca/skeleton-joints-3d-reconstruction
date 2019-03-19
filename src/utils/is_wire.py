import socket
from enum import Enum
from is_wire.core import Subscription, Message, Logger


class RequestManager:
    def __init__(self, channel, max_requests, log_level=Logger.INFO):

        self._channel = channel
        self._subscription = Subscription(self._channel)

        self._log = Logger(name='RequestManager')
        self._log.set_level(level=log_level)

        self._max_requests = max_requests
        self._requests = {}

    def can_request(self):
        return len(self._requests) < self._max_requests

    def all_received(self):
        return len(self._requests) == 0

    def request(self, content, topic, timeout_ms, metadata):

        if not self.can_request():
            raise Exception("Can't request more than {}. Use 'RequestManager.can_request' "
                            "method to check if you can do requests.")

        msg = Message(content=content)
        msg.topic = topic
        msg.reply_to = self._subscription
        msg.timeout = timeout_ms / 1000.0

        self._log.debug("[Sending] metadata={}, cid={}", metadata, msg.correlation_id)
        self._publish(msg, metadata)

    def consume(self, timeout=1.0):
        received = None

        # wait for new message
        try:
            msg = self._channel.consume(timeout=timeout)

            if msg.status.ok() and msg.has_correlation_id():
                cid = msg.correlation_id

                if cid in self._requests:
                    received = (msg, self._requests[cid]["metadata"])
                    del self._requests[cid]

        except socket.timeout:
            pass

        # check for timeouted requests
        for cid in self._requests.keys():
            timeouted_msg = self._requests[cid]["msg"]

            if timeouted_msg.deadline_exceeded():
                msg = Message()
                msg.body = timeouted_msg.body
                msg.topic = timeouted_msg.topic
                msg.reply_to = self._subscription
                msg.timeout = timeouted_msg.timeout

                metadata = self._requests[cid]["metadata"]

                del self._requests[cid]

                self._log.debug("[Retring] metadata={}, cid={}", metadata, msg.correlation_id)
                self._publish(msg, metadata)

        return received

    def _publish(self, msg, metadata):
        self._channel.publish(message=msg)
        self._requests[msg.correlation_id] = {
            "msg": msg,
            "metadata": metadata,
        }
