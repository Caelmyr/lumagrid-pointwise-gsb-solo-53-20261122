import os

from waitress import serve

from .http import create_app


if __name__ == "__main__":
    serve(
        create_app(),
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        threads=4,
        channel_timeout=60,
        max_request_body_size=32 * 1024 * 1024,
        clear_untrusted_proxy_headers=True,
    )
