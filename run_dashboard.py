"""Launch the OmniGamer Web Dashboard with live stream and controls."""
import argparse

from omnigamer.personality import configure_stdio

configure_stdio()

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniGamer Web Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host/interface to bind (default: 127.0.0.1)")
    args = parser.parse_args()
    print(f"\nStarting OmniGamer Live Dashboard at http://{args.host}:8765\n")
    uvicorn.run("omnigamer.dashboard.app:app", host=args.host, port=8765, reload=False)
