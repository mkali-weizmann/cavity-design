import serial
import csv

# Open the serial port (replace with your ESP32 COM port)
ser = serial.Serial('COM8', 115200)  # Replace 'COM3' with the correct port

# Open a CSV file to write the data
with open('sensor_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Timestamp', 'MQ9', 'MQ135'])  # Write header row

    for i in range(100):
        line = ser.readline().decode('utf-8').strip()  # Read a line from the ESP32
        if line.startswith("Timestamp"):  # Skip the header line
            continue
        if line:  # If the line is not empty
            data = line.split(',')  # Split by commas
            writer.writerow(data)  # Write the data to the CSV file

ser.close()  # Close the serial connection