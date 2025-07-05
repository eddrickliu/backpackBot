from backpack_exchange_sdk.public import PublicClient

public_client = PublicClient()

# Get all supported assets
assets = public_client.get_assets()
print(assets)

# Get ticker information for a specific symbol
ticker = public_client.get_ticker('SOL_USDC')
print(ticker)
# Get ticker information for a specific symbol
ticker = public_client.get_depth('SOL_USDC')
print(ticker)

print(public_client.get_market("SOL_USDC"))
# print(public_client.get_market("SOL/USDC"))
# print(public_client.get_market("SOL-PERP"))