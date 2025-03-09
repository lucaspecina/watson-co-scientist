#!/usr/bin/env python3
"""
Script to start the Raul Co-Scientist API server.
"""

import sys
import argparse
import uvicorn

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Start the Raul Co-Scientist API server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    # Start the uvicorn server
    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    sys.exit(main())