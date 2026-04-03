def generate_safe_signature(account, proxy_wallet: str, to: str, value: int, data: bytes, operation: int, safe_tx_gas: int, base_gas: int, gas_price: int, gas_token: str, refund_receiver: str, nonce: int, chain_id: int) -> bytes:
    from eth_account.messages import encode_typed_data

    domain = {
        "verifyingContract": proxy_wallet,
        "chainId": chain_id,
    }
    types = {
        "EIP712Domain": [
            {"name": "verifyingContract", "type": "address"},
            {"name": "chainId", "type": "uint256"}
        ],
        "SafeTx": [
            {"name": "to", "type": "address"},
            {"name": "value", "type": "uint256"},
            {"name": "data", "type": "bytes"},
            {"name": "operation", "type": "uint8"},
            {"name": "safeTxGas", "type": "uint256"},
            {"name": "baseGas", "type": "uint256"},
            {"name": "gasPrice", "type": "uint256"},
            {"name": "gasToken", "type": "address"},
            {"name": "refundReceiver", "type": "address"},
            {"name": "nonce", "type": "uint256"}
        ]
    }
    
    # EOA is 1.1.1 version of Safe typically on Polymarket.
    message = {
        "to": to,
        "value": value,
        "data": "0x" + data.hex() if isinstance(data, bytes) else data,
        "operation": operation,
        "safeTxGas": safeTxGas,
        "baseGas": baseGas,
        "gasPrice": gasPrice,
        "gasToken": gas_token,
        "refundReceiver": refund_receiver,
        "nonce": nonce
    }
    
    # eth_account EIP-712 requires `message_types` without EIP712Domain
    # But wait, wait: eth_account.messages.encode_typed_data takes standard JSON.
    eip712_json = {
        "types": types,
        "primaryType": "SafeTx",
        "domain": domain,
        "message": message
    }
    
    signable = encode_typed_data(full_message=eip712_json)
    signed = account.sign_message(signable)
    
    # Gnosis Safe requires signatures array: bytes32 r, bytes32 s, uint8 v
    # Packed properly. eth_account signature is usually v,r,s. Safe requires r,s,v.
    r = signed.r.to_bytes(32, byteorder='big')
    s = signed.s.to_bytes(32, byteorder='big')
    v = signed.v
    return r + s + bytes([v])
