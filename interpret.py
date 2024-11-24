import os
import json
from scapy.all import rdpcap, TCP

def concatenate_tcp_payloads(packets):
    concatenated_payloads = ""
    for packet in packets:
        if TCP in packet and packet[TCP].payload:
            payload = bytes(packet[TCP].payload).hex()
            concatenated_payloads += payload
    return concatenated_payloads

def process_file(file_path):
    packets = rdpcap(file_path)
    
    # Concatenate payloads
    hex_string = concatenate_tcp_payloads(packets)
    
    # Decode hex string
    try:
        bytes_payload = bytes.fromhex(hex_string)
        human_readable = bytes_payload.decode('utf-8', errors='replace')
    except ValueError:
        human_readable = ""
    
    # Remove newlines and extra whitespaces
    human_readable = human_readable.replace('\n', '').replace(' ', '').strip()
    
    return human_readable

f = open(os.path.join(os.sep, "usr", "src", "app", "InputData", "train", "train.json"), "w")
f.write("{\n")

i = 0
for dirs, subdirs, files in os.walk(os.path.join(os.sep, "usr", "src", "app","InputData", "train")):
    for file in files:
        if file == "train.json":
            continue
        input_file_path = os.path.join(os.sep, "usr", "src", "app", "InputData", "train", file)
        output_file_path = os.path.join(os.sep, "usr", "src", "app", "InputData", "train", "train.json")
        human_readable = process_file(input_file_path)
        f = open(os.path.join(os.sep, "usr", "src", "app", "InputData", "train", "train.json"), "a")
        f.write(f"\n \"{i}\" : ")
        f.write(human_readable)
        if(len(files) - 2 != i):
            f.write(",\n")
        i = i + 1
        if i > 200:
            break

f = open(os.path.join(os.sep, "usr", "src", "app", "InputData", "train", "train.json"), "a")
f.write("}\n")

f = open(os.path.join(os.sep, "usr", "src", "app", "InputData", "test", "test.json"), "w")
f.write("{\n")

i = 0
for dirs, subdirs, files in os.walk(os.path.join(os.sep, "usr", "src", "app", "InputData", "test")):
    for file in files:
        if file == "test.json":
            continue
        input_file_path = os.path.join(os.sep, "usr", "src", "app", "InputData", "test", file)
        output_file_path = os.path.join(os.sep, "usr", "src", "app", "InputData", "test", "test.json")
        human_readable = process_file(input_file_path)
        f = open(os.path.join(os.sep, "usr", "src", "app", "InputData", "test", "test.json"), "a")
        f.write(f"\n \"{i}\" : ")
        f.write(human_readable)
        if(len(files) - 2 != i):
            f.write(",\n")
        i = i + 1

f = open(os.path.join(os.sep, "usr", "src", "app", "InputData", "test", "test.json"), "a")
f.write("}\n")