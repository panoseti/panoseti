import json
import socket


def send_fixed_packet(sock, data_dict, packet_size=1024):
    """
    Serializes a dictionary, pads it to a fixed size, and sends it over a socket.

    Args:
        sock (socket.socket): The socket to send data through.
        data_dict (dict): The dictionary to be sent.
        packet_size (int): The exact size of the packet to send in bytes.

    Raises:
        ValueError: If the serialized dictionary is larger than the packet_size.
    """
    # 1. Serialize the dictionary to a JSON formatted string
    json_str = json.dumps(data_dict)

    # 2. Encode the string into bytes using UTF-8
    encoded_data = json_str.encode('utf-8')

    # 3. Check if the encoded data exceeds the allowed packet size
    if len(encoded_data) > packet_size:
        raise ValueError(f'Data size ({len(encoded_data)} bytes) exceeds fixed packet size ({packet_size} bytes)')

    # 4. Pad the data with spaces to make it exactly packet_size bytes long
    padded_data = encoded_data.ljust(packet_size, b' ')

    # 5. Send the complete fixed-size packet
    sock.sendall(padded_data)
    print(f"Sent a fixed-size packet of {len(padded_data)} bytes.")


def recv_fixed_packet(sock, packet_size=1024):
    """
    Receives a fixed-size packet from a socket and deserializes it into a dictionary.

    Args:
        sock (socket.socket): The socket to receive data from.
        packet_size (int): The exact size of the packet to receive in bytes.

    Returns:
        dict: The reconstructed dictionary from the received packet.

    Raises:
        ConnectionError: If the connection is lost before the full packet is received.
    """
    received_bytes = b''
    # 1. Loop until the entire fixed-size packet is received
    while len(received_bytes) < packet_size:
        # Calculate how many more bytes are needed
        bytes_to_read = packet_size - len(received_bytes)
        chunk = sock.recv(bytes_to_read)

        if not chunk:
            # This occurs if the connection is closed by the peer
            raise ConnectionError('Socket connection closed before receiving the full packet.')

        received_bytes += chunk

    print(f"Received a fixed-size packet of {len(received_bytes)} bytes.")

    # 2. Strip any trailing space characters used for padding
    stripped_data = received_bytes.rstrip(b' ')

    # 3. Decode the bytes back into a JSON string
    json_str = stripped_data.decode('utf-8')

    # 4. Deserialize the JSON string to reconstruct the dictionary
    data_dict = json.loads(json_str)

    return data_dict


# --- Example Usage ---
if __name__ == '__main__':
    # Define the fixed packet size for communication
    FIXED_PACKET_SIZE = 256

    # Create a pair of connected sockets for a local demonstration
    server_socket, client_socket = socket.socketpair()

    # The dictionary we want to send
    message_to_send = {
        "status": "ok",
        "code": 200,
        "payload": {"item": "test", "value": 123}
    }

    try:
        # --- Sending side (Client) ---
        print("--- Sending ---")
        send_fixed_packet(client_socket, message_to_send, packet_size=FIXED_PACKET_SIZE)

        # --- Receiving side (Server) ---
        print("\n--- Receiving ---")
        received_message = recv_fixed_packet(server_socket, packet_size=FIXED_PACKET_SIZE)

        # --- Verification ---
        print("\n--- Verification ---")
        print(f"Original data:  {message_to_send}")
        print(f"Received data:  {received_message}")
        assert message_to_send == received_message
        print("Success: The received data accurately matches the original data.")

    except ValueError as e:
        print(f"Error during sending: {e}")
    except ConnectionError as e:
        print(f"Error during receiving: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        # Always close the sockets to release system resources
        server_socket.close()
        client_socket.close()
