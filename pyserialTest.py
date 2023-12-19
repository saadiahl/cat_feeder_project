import serial
import time

# Establish a serial connection
ser = serial.Serial('COM3', 9600)  # Replace 'COM3' with the correct port
time.sleep(2)  # wait for the serial connection to initialize

def send_float_to_arduino(value):
    ser.write(f"{value}\n".encode())

def receive_float_from_arduino():
    while ser.in_waiting:
        try:
            data = ser.readline().decode().strip()
            return float(data)
        except ValueError:
            print("Received non-float data")
            return None

# Example Usage
send_float_to_arduino(3.14)
received_value = receive_float_from_arduino()
if received_value is not None:
    print(f"Received: {received_value}")
