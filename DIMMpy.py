import aotools as ao
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift
from astropy.io import fits
from datetime import datetime
import copy

class Reader:
	def __init__(self):
		self.readData = None

	def ReadFits(self, fileName):
		hduList = fits.open(fileName)
		self.readData = []
		for hdu in hduList:
			self.readData.append(hdu.data)
		self.readData = np.array(self.readData)

	def ReadSers(self, filename):
		file = np.fromfile(filename, dtype=np.int16)
		data = file[489:]
		self.readData = np.reshape(data, [100, 480, 640])

class Writer:
	def __init__(self):
		self.writeData = None

	def WriteFits(self, fileName):
		fileName = fileName[:-3]+'FIT'
		for frame in self.writeData:
			fits.append(fileName, frame)

class DIMM(Writer):
	def __init__(self, r0, L0, radius):
		super().__init__()
		self.nx_size = 512
		self.shift = radius+2
		self.leftPupilMask = ao.circle(radius, self.nx_size, (-self.shift, 0))
		self.rightPupilMask = ao.circle(radius, self.nx_size, (self.shift, 0))
		self.apeture = 0.12
		self.pxlScale = 0.05/(2*radius) # scale of phase screen pixels
		self.upperphaseScreen = ao.turbulence.infinitephasescreen. PhaseScreenKolmogorov(self.nx_size, self.pxlScale, r0, L0)
		self.psf = np.zeros((4536, 4536))
		self.image = np.zeros((4536, 4536))
		self.padWidth = PadWidth(self.pxlScale, self.nx_size)

	def ShowPhaseScreen(self):
		plt.figure()
		plt.imshow(self.upperphaseScreen.scrn)
		plt.colorbar()
		plt.show()

	def ShowPSF(self):
		plt.figure()
		plt.imshow(self.psf[:, :])
		plt.colorbar()
		plt.show()

	def ShowImage(self):
		plt.figure()
		plt.imshow(self.image[:, :])
		plt.colorbar()
		plt.show()

	def CreatePSF(self):
		wavelength = 550 * 10 ** (-9)
		distance = 10000
		compWavefront = np.exp(1j*self.upperphaseScreen.scrn)

		propogatedScreen = ao.opticalpropagation.angularSpectrum(compWavefront, wavelength, self.pxlScale, self.pxlScale, distance)

		leftMasked = np.pad((self.leftPupilMask*propogatedScreen), self.padWidth)
		leftImage = np.abs(ao.ft2(leftMasked, self.pxlScale))**2

		rightMasked = np.pad(self.rightPupilMask*propogatedScreen, self.padWidth)
		rightImage = np.abs(ao.ft2(rightMasked, self.pxlScale))**2

		self.psf = shift(leftImage, (0, -49)) + shift(rightImage, (0, 49))

		scale = 2 * EventsExpected() / np.sum(self.psf)

		self.psf = self.psf*scale
		self.psf = BinPixels(self.psf, 3)

	def CreateImage(self):
		self.CreatePSF()
		self.image = np.random.poisson(self.psf)

	def CreateImageStack(self, N):
		self.writeData = []
		for i in range(N):
			self.CreateImage()
			self.writeData.append(self.image)
			self.upperphaseScreen.add_row()
		self.writeData = np.array(self.writeData)
		plt.imshow(self.writeData[5])
		plt.show()

	def CreateRandomImageStack(self, N):
		self.writeData = []
		pos = np.random.randint(-1012, 1012, 2)
		time = datetime.now()
		Name = time.strftime("%H%M%S")
		for i in range(N):
			print(i)
			self.CreateImage()
			largeImage = np.pad(self.image, 1024)
			movedImage = shift(largeImage, pos)
			movedImage = np.round(np.random.normal(movedImage, 2.4)) + 5 # adding read noise with a bias of 5
			self.writeData.append(movedImage)
			for i in range(50):
				self.upperphaseScreen.add_row()
		self.WriteFits(Name+".FIT")

class Stripper(Reader, Writer):
	def __init__(self, stripSize):
		super().__init__()
		self.stripSize = stripSize

	def Strip(self, fileName, fileFormat = 'SER'):
		if fileFormat not in ['SER', 'FIT']:
			raise Exception(str(fileFormat)+" is not a supported file extension, please enter 'FIT' or 'SER'")
		if fileFormat=='FIT':
			self.ReadFits(fileName)
		else:
			self.ReadSers(fileName)
		medianX, medianY = FindSpot(self.readData[0], 19)
		self.writeData = []
		for frame in self.readData:
			self.writeData.append(frame[medianX-self.stripSize:medianX+self.stripSize, medianY-self.stripSize:medianY+self.stripSize])
		self.writeData = np.array(self.writeData)
		self.WriteFits("stripped_"+fileName)

class Analyser(Reader):
	def __init__(self):
		super().__init__()
		self.DifferentialArray = np.array([])
		self.centerSpotIntensity = np.array([])
		self.outerSpotIntensity = np.array([])
		self.backgroundMax = 0

	def SubtractBackground(self, size):
		backgroundArray = self.readData[:,:size,:size]
		sigma = np.std(backgroundArray)
		self.backgroundMax = np.round(np.mean(backgroundArray) + 4*sigma) # covers 99.9% of background points if background is normally distributed
		self.readData = self.readData - self.backgroundMax
		self.readData = np.where(self.readData<0, 0, self.readData)
		print('sigma: ', sigma)
		print('background max: ', self.backgroundMax)

	def FindDifferentialDistance(self, windowSize, D, d, wavelength, pixelScale):
		imageShape = self.readData[0].shape
		centralMask = ao.circle(15, imageShape[0])
		outerMask = ao.circle(32, imageShape[0])-ao.circle(18, imageShape[0])
		spotLotatorFrames = np.sum(self.readData[:3], 0) # sums 3 consecutive frames, so we can ensure there is a bright spot in each location.
		centerLocX, centerLocY = FindSpot(spotLotatorFrames*centralMask, 9)
		outerLocX, outerLocY = FindSpot(spotLotatorFrames*outerMask, 9)
		centerImages = self.readData[:,centerLocX-windowSize:centerLocX+windowSize,centerLocY-windowSize:centerLocY+windowSize]
		outerImages = self.readData[:,outerLocX-windowSize:outerLocX+windowSize,outerLocY-windowSize:outerLocY+windowSize]

		deleteArray = []
		threshold = 20*self.backgroundMax
		for i in range(len(centerImages)):
			if np.sum(centerImages[i])<threshold:
				deleteArray.append(i)
			elif np.sum(outerImages[i])<threshold:
				deleteArray.append(i)
		centerImages = np.delete(centerImages, deleteArray, 0)
		outerImages = np.delete(outerImages, deleteArray, 0)
		print(deleteArray)

		centerCentroids = ao.centre_of_gravity(centerImages)
		outerCentroids = ao.centre_of_gravity(outerImages)
		DifferenceArray = (centerCentroids - outerCentroids) * pixelScale
		sigmaL = np.std(DifferenceArray[1])
		sigmaT = np.std(DifferenceArray[0])
		#print(DifferenceArray[1])
		print(sigmaL)
		r0L = ((sigmaL**2)/(2*wavelength**2 * (0.179*D**(-1/3) - 0.0968 * d**(-1/3))))**(-3/5)
		print(r0L)

	def FindAtmoParameters(self):
		return "test"

"""General Functions"""

def EventsExpected(magnitude = 2, quantumEfficiency = 0.6, radius = 2.5, exposure = 0.002, bandWidth = 800):
	return 1000 * 10**(-magnitude/2.5) * quantumEfficiency * np.pi * radius**2 * exposure * bandWidth

def PadWidth(pxlScaleIn, nx_size, wavelength = 500*10**(-9), pxlScaleOut = 17.7/4656, S = 1/480):
	return int((((wavelength / (S * pxlScaleOut * pxlScaleIn)) * 9)-nx_size)/2)

def BinPixels(array, n):
	binnedShape = (int(np.trunc(array.shape[0] / n)), int(np.trunc(array.shape[1] / n)))
	binnedArray = np.zeros(binnedShape)
	for i in range(binnedShape[0]):
		for j in range(binnedShape[1]):
			binnedArray[i,j] = np.sum(array[n*i:n*i+n, n*j:n*j+n])
	return binnedArray

def FindSpot(data, N):
	data = copy.deepcopy(data)
	XIndexArray = np.empty(N)
	YIndexArray = np.empty(N)
	for i in range(N):
		index = np.unravel_index(np.argmax(data), data.shape)
		data[index] = 0
		XIndexArray[i] = index[0]
		YIndexArray[i] = index[1]
	return int(np.median(XIndexArray)), int(np.median(YIndexArray))
