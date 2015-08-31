import urllib
import os
import os.path
import gzip

if not os.path.exists('mnist'):
	os.mkdir('mnist')
urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', './mnist/train-images-idx3-ubyte.gz')
urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', './mnist/train-labels-idx1-ubyte.gz')
urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', './mnist/t10k-images-idx3-ubyte.gz')
urllib.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', './mnist/t10k-labels-idx1-ubyte.gz')

def extract(name):
	inF = gzip.GzipFile(name, 'rb')
	s = inF.read()
	inF.close()

	outF = file(name[:-3], 'wb')
	outF.write(s)
	outF.close()
        os.remove(name)

extract('./mnist/train-images-idx3-ubyte.gz')
extract('./mnist/train-labels-idx1-ubyte.gz')
extract('./mnist/t10k-images-idx3-ubyte.gz')
extract('./mnist/t10k-labels-idx1-ubyte.gz')
