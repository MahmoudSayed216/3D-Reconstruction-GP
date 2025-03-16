# this method is a very accurate but not an exact calculation
from model import decoder, encoder, merger, refiner
d = decoder.Decoder()
e = encoder.Encoder("cpu", False)
m = merger.Merger(0.2)
r = refiner.Refiner(0.2, False)

sum_params = lambda M: sum(p.numel() for p in M.parameters()) 

parameters = sum([sum_params(M) for M in [e, d, m, r]])


size_in_mb = parameters>>18 


print(size_in_mb)