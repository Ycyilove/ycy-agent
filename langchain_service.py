"""Compatibility entrypoint for the modular backend application.

Use ``backend.app:app`` for new deployments. This facade keeps the historical
``langchain_service`` imports and ``python langchain_service.py`` command valid.
"""

import time
_t0 = time.time()
print(f"[boot] start: {time.strftime('%H:%M:%S')}", flush=True)

from backend.app import *  # noqa: F401,F403
from backend.app import app

print(f"[boot] imports done: {time.strftime('%H:%M:%S')} "
      f"(+{time.time()-_t0:.1f}s)", flush=True)

if __name__ == "__main__":
    print(f"[boot] entering uvicorn: {time.strftime('%H:%M:%S')} "
        f"(+{time.time()-_t0:.1f}s)", flush=True)
    import uvicorn
    uvicorn.run(
        app,
        host="localhost",
        port=8000,
        timeout_keep_alive=300,       # keep-alive 5 分钟
        timeout_graceful_shutdown=30, # 优雅关闭 30 秒
    )