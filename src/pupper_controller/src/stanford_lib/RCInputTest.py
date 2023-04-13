import hid

gamepad = hid.device()
#gamepad.open(0x0483, 0x5720)
for device in hid.enumerate():
    if device['product_string'] == 'FrSky Simulator':
        gamepad.open(device['vendor_id'], device['product_id'])
        break

gamepad.set_nonblocking(True)

while True:
    report = gamepad.read(64)
    if report:
        print([r - 256 if r > 127 else r for r in report])