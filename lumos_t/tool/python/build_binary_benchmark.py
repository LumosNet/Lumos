import struct

a = [10, 20, -23]
b = 12
fw = open("file_name.bin", "wb")
fw.write(b.to_bytes(4, 'little'))
for i in a:
    fw.write(i.to_bytes(2, 'little'))
fw.close()
