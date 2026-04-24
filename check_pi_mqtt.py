import asyncio
import aiomqtt
import json
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def check_mqtt():
    pi_ip = "192.168.0.100"
    print(f"Attempting to connect to Pi's MQTT broker at {pi_ip}...")
    try:
        async with aiomqtt.Client(hostname=pi_ip, port=1883, timeout=5) as client:
            print("Successfully connected to Pi's MQTT broker!")
            await client.subscribe("vitals/#")
            print("Subscribed to 'vitals/#' topics. Waiting for messages for 10 seconds...")
            
            async with asyncio.timeout(10):
                async for message in client.messages:
                    print(f"Received message on {message.topic}: {message.payload.decode()}")
    except asyncio.TimeoutError:
        print("No messages received in 10 seconds.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_mqtt())
