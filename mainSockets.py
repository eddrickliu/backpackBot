from backpack_exchange_sdk.websocket import WebSocketClient
from datetime import datetime

# Initialize WebSocket client
# ws_client = WebSocketClient()  # For public streams only
# todo: replace with api keys
ws_client = WebSocketClient(api_key='', secret_key='')  # For private streams

# Define callback functions
def handle_book_ticker(data):
    """Handle book ticker updates"""
    print("\n=== Book Ticker Update ===")
    print(f"Symbol: {data['s']}")
    print(f"Best Ask: {data['a']} (Quantity: {data['A']})")
    print(f"Best Bid: {data['b']} (Quantity: {data['B']})")
    print(f"Time: {datetime.fromtimestamp(data['E']/1000000).strftime('%Y-%m-%d %H:%M:%S.%f')}")

def handle_trades(data):
    """Handle trade updates"""
    print("\n=== Trade Update ===")
    print(f"Symbol: {data['s']}")
    print(f"Price: {data['p']}")
    print(f"Quantity: {data['q']}")
    print(f"Trade Type: {'Maker' if data['m'] else 'Taker'}")
    print(f"Trade ID: {data['t']}")

# Subscribe to public streams
ws_client.subscribe(
    streams=["bookTicker.SOL_USDC"],  # Book ticker stream
    callback=handle_book_ticker
)

ws_client.subscribe(
    streams=["bookTicker.SOL_USDT"],  # Book ticker stream
    callback=handle_book_ticker
)

# ws_client.subscribe(
#     streams=["bookTicker.BTC_USDC"],  # Book ticker stream
#     callback=handle_book_ticker
# )

# ws_client.subscribe(
#     streams=["bookTicker.ETH_USDC"],  # Book ticker stream
#     callback=handle_book_ticker
# )

# Subscribe to private streams (requires authentication)
ws_client.subscribe(
    streams=["account.orderUpdate.SOL_USDC"],  # Order updates stream
    callback=handle_trades,
    is_private=True
)

# Keep the connection alive
import time
while True:
    time.sleep(1)