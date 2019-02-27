#!/usr/bin/env python3

import time
import serial
import sys
from PIL import Image

sync = b"\xaa\x0d\x00\x00\x00\x00"
ack_prefix = b"\xaa\x0e"

def print_bytes(bs):
	for b in bs:
		print("%02x" % b, end=" ")

def is_ack(response, cmd):
	return response[:2] == ack_prefix and response[2] == cmd[1]

def send_cmd(port, cmd):
	port.write(cmd)
	response = port.read(6)
	if is_ack(response, cmd):
		return True
	else:
#		print("did not get ACK, response: [", end="")
#		print_bytes(response)
#		print("]")
		return False

def synchronise(port):
	pause = 0.005
	for i in range(60):
		if send_cmd(port, sync):
			response = port.read(6)
			if response == sync:
				port.write(ack_prefix + b"\x0d\x00\x00\x00")
				return True
			else:
				return False
		else:
			time.sleep(pause)
			pause += 0.001
	return False

with serial.Serial("/dev/ttyUSB0", timeout=3, baudrate=115200) as port:
	if not synchronise(port):
		print("failed to sync")
		sys.exit(1)

	time.sleep(1)

	print("sending INITIAL...", end=" ")
	if send_cmd(port, b"\xaa\x01\x00\x03\x09\x00"):
		print("success")
	else:
		print("failed")
		sys.exit(1)

	print("sending GET_PICTURE")
	if send_cmd(port, b"\xaa\x04\x02\x00\x00\x00"):
		print("success")
	else:
		print("failed")
		sys.exit(1)

	response = port.read(6)
	if response[:2] != b"\xaa\x0a":
		print("did not get DATA header")
		sys.exit(1)

	length = int.from_bytes(response[3:], "big")
	print("image size:", length, "bytes")

	image_bytes = port.read(length)
	print("received", len(image_bytes), "bytes")
	port.write(b"\xaa\x0e\x0a\x00\x01\x00")

	image = Image.frombytes("L", (128, 128), image_bytes)
	image.show()
	image.save("img.png")
