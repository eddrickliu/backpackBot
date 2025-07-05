from enums import RequestEnums
from backpack_exchange_sdk.authenticated import AuthenticationClient
from dotenv import load_dotenv
import os

def main():
    print("Hello from main!")# Get account balances

    client = AuthenticationClient(os.getenv("API_KEY"), os.getenv("SECRET_KEY"))
    balances = client.get_balances()
    print(balances)

    # # Request a withdrawal
    # response = client.request_withdrawal('xxxxaddress', 'Solana', '0,1', 'Sol')
    # print(response)
    try:

        # Execute a limit order

        print(
            client.execute_order(
                RequestEnums.OrderType.LIMIT.value,
                RequestEnums.Side.ASK.value,
                "SOL_USDC",
                postOnly=True,
                clientId=12345,
                price="152",
                quantity="0.01",
                timeInForce=RequestEnums.TimeInForce.GTC.value,
                # selfTradePrevention="decrementAndCancel",
            )
        )
    except Exception as e:
        print(f"Error executing order: {e}")

if __name__ == "__main__":
    main()