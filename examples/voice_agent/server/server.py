import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from bot_websocket import run_bot_websocket_server
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

load_dotenv(override=True)
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", 8765))
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 8766))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles FastAPI startup and shutdown."""
    yield  # Run app


# Initialize FastAPI app with lifespan manager
app = FastAPI(lifespan=lifespan)

# Configure CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted")
    try:
        # TODO: [heh] Implement FastAPI websocket endpoint
        # await run_bot_fastapi_server(websocket)
        raise NotImplementedError("FastAPI websocket endpoint is not implemented")
    except Exception as e:
        print(f"Exception in run_bot: {e}")


@app.post("/connect")
async def bot_connect(request: Request) -> Dict[Any, Any]:
    print("Received /connect request")
    # Use the host that the client connected to (from the request)
    server_host = request.url.hostname or request.headers.get("host", "").split(":")[0]
    ws_url = f"ws://{server_host}:{WEBSOCKET_PORT}"
    print(f"Returning WebSocket URL: {ws_url}")
    return {"ws_url": ws_url}


async def main():
    """Main function to run both websocket server and FastAPI server concurrently."""
    logger.info(f"Starting servers - WebSocket on port {WEBSOCKET_PORT}, FastAPI on port {FASTAPI_PORT}")
    tasks = []
    try:
        # Start websocket server
        tasks.append(run_bot_websocket_server(host=SERVER_HOST, port=WEBSOCKET_PORT))

        # Start FastAPI server
        config = uvicorn.Config(app, host=SERVER_HOST, port=FASTAPI_PORT)
        server = uvicorn.Server(config)
        tasks.append(server.serve())

        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Tasks cancelled (probably due to shutdown).")


if __name__ == "__main__":
    asyncio.run(main())
