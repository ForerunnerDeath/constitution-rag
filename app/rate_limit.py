from collections import defaultdict, deque
from time import perf_counter


class RateLimiter:
    def __init__(self, *, max_requests: int, window_seconds: float = 60.0) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be greater than zero")

        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero")

        self.max_requests = max_requests
        self.window_seconds = window_seconds

        self._requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client_id: str) -> bool:
        now = perf_counter()

        requests = self._requests[client_id]

        window_start = now - self.window_seconds

        while requests and requests[0] <= window_start:
            requests.popleft()

        if len(requests) >= self.max_requests:
            return False

        requests.append(now)

        return True
