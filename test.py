import Karanir



mlp = Karanir.dklearn.FCBlock(1,2,3)

gcv = Karanir.dklearn.GraphConvolution(1,2,3)

print(Karanir.math.rotate(4,4))

print(gcv(1))

print(Karanir.utils.test_read())

print(mlp(5))