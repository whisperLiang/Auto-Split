
import base64
from xmlrpc.server import SimpleXMLRPCServer
from bitstruct import *

def output(xx_encoded):
    # Cloud side ----
    num_bits=3
    len_symbols=9
    # C1. decode utf-8 string to bytes
    xx_bytes = base64.b64decode(xx_encoded)
    # C2. decode bytes to array/list
    from bitstruct import unpack
    dtype = 's{}'.format(num_bits) * len_symbols
    recovered_symbols = unpack(dtype, xx_bytes)
    print(recovered_symbols)


server = SimpleXMLRPCServer(('localhost', 1234), allow_none=True)
print('Serving localhost 1234 ...')
server.register_function(output)
server.serve_forever()
