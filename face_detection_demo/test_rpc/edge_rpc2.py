# from bitstruct import *
# import xmlrpc.client
# import base64
# device = xmlrpc.client.ServerProxy("http://localhost:1234/RPC2", allow_none=True)
# symbols = [-4, -3, -2, -1, 0, 1, 2, 3, 3]
# dtype = 's3'*len(symbols)
# p_x = pack(dtype, *symbols)
# # Converting binary data to hex string.
# hex_str = p_x.hex()
# # Transferring the string over the network.
# p_y = device.output(hex_str)
# print(type(p_y))
