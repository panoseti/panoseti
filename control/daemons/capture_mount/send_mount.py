import json
import socket
import time

# --- Client Configuration ---
# IMPORTANT: Replace with the server's actual IP address
SERVER_HOST = '10.146.200.1'
SERVER_PORT = 60010
PACKET_SIZE = 1024  # Must be <= server's BUFFER_SIZE

# Data to send
message_to_send = {
    "mount_id": "client_" + str(int(time.time() * 1000))[-4:],  # Unique-ish ID
    "message": "Hello from a UDP client!",
    "data_value": 42
}



# 1. Create a UDP socket
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
    try:
        # --- Prepare the data ---
        # 1. Serialize dictionary to JSON string
        json_str = json.dumps(message_to_send)

        # 2. Encode to bytes
        encoded_data = json_str.encode('utf-8')

        # 3. Check size and pad to fixed length
        if len(encoded_data) > PACKET_SIZE:
            raise ValueError(f"Data size ({len(encoded_data)}) exceeds packet size ({PACKET_SIZE})")

        padded_data = encoded_data.ljust(PACKET_SIZE, b' ')

        # 2. Send the data to the server's address and port
        #    No connect() call is needed for UDP.
        client_socket.sendto(padded_data, (SERVER_HOST, SERVER_PORT))
        print(f"Sent packet to {SERVER_HOST}:{SERVER_PORT}")
        print(f"Data: {message_to_send}")

        # --- (Optional) Wait for a response from the server ---
        # Set a timeout so the client doesn't wait forever if the response is lost
        client_socket.settimeout(5.0)  # 5-second timeout

        response_bytes, server_address = client_socket.recvfrom(PACKET_SIZE)
        response_dict = json.loads(response_bytes.decode('utf-8'))

        print(f"\nReceived response from server {server_address}:")
        print(f"Response data: {response_dict}")

    except socket.timeout:
        print("No response from server; request may have been lost.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

