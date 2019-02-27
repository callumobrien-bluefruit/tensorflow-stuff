#!/usr/bin/env python3

import time
import serial

sync_cmd = b'\xaa\x0d\x00\x00\x00\x00'
ack_prefix = b'\xaa\x0e'

def send_cmd(port, cmd):
	port.write(sync_cmd)
	response = port.readline()
	return response[:2] == ack_prefix

def synchronise(port):
	pause = 0.005
	for i in range(60):
		if send_cmd(port, sync_cmd):
			return True
		else:
			time.sleep(pause)
			pause += 0.001
	return False

with serial.Serial('/dev/ttyUSB0', timeout=1) as port:
	if synchronise(port):
		print('synced')
	else:
		print('failed to sync')

	#time.sleep(1)
	#send_cmd(b'\xaa\x01\x00\x03\x09\x00')
