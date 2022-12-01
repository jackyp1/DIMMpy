from datetime import datetime

import DIMMpy as dp
import Graphing as gp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def analCode():
	Analysis = dp.Analyser()
	Analysis.ReadFits('stripped_134855.FIT')
	#Analysis.readData = Analysis.readData/16
	Analysis.SubtractBackground(20)
	Analysis.FindDifferentialDistance(15, 0.05, 0.30, 550*10**(-9), (17.7/4656)/480)

	test = dp.Writer()
	test.writeData = Analysis.outerImages
	test.WriteFits(fileName="test.FIT")

def simCode():
	sim = dp.DIMM(0.12, 2, 30)
	sim.CreatePsfStack(100)
	# sim.CreateRandomImageStack(100)

def stripCode():
	stripper = dp.Stripper(60)
	stripper.Strip("134855.FIT", "FIT")

def BackgroundAnal():
	file = dp.Reader()
	file.ReadFits('stripped_2022-10-26-1856_0-CapObj.FIT')
	data = file.readData[0:30, 0:30]/16
	data = data.astype('int16')
	gp.PlotHistogram('background hist', data.flatten(), 'Pixel Value', 'Frequency')

def FindMeasuredR0():
	r0Array = np.linspace(0.02, 0.3, 5)
	strip = dp.Stripper(40)
	analyser = dp.Analyser()
	r0Dict = {'L': {}, 'T':{}}
	r0LErrorArray = []
	r0TErrorArray = []
	nPerPoint = 10
	for r0 in r0Array:
		print(r0)
		r0Dict['L'][r0] = []
		r0Dict['T'][r0] = []
		for i in range(nPerPoint):
			sim = dp.DIMM(r0, 2, 30)
			sim.CreatePsfStack(1000)
			strip.readData = np.pad(sim.writeData, 40)[40:-40]
			strip.Strip()
			analyser.readData = strip.writeData
			r0L, r0T = analyser.FindDifferentialDistance(15, 0.05, 0.30, 550 * 10 ** (-9), (17.7 / 4656) / 480)
			r0Dict['L'][r0].append(r0L)
			r0Dict['T'][r0].append(r0T)
	r0Array = r0Dict['L'].keys()
	r0LArray = []
	r0TArray = []
	for r0 in r0Array:
		r0LArray.append(np.mean(r0Dict['L'][r0]))
		r0LErrorArray.append(np.std(r0Dict['L'][r0])/np.sqrt(nPerPoint))
		r0TArray.append(np.mean(r0Dict['T'][r0]))
		r0TErrorArray.append(np.std(r0Dict['T'][r0]) / np.sqrt(nPerPoint))
	plt.figure(figsize=(7,5))
	plt.figure(1).add_axes((0,0,1,0.8))
	plt.errorbar(r0Array, r0LArray, yerr=r0LErrorArray, marker='x', mec='black', capsize=2, ecolor='black', elinewidth=0.5, label='r0L')
	plt.errorbar(r0Array, r0TArray, yerr=r0TErrorArray, marker='o', mec='black', capsize=2, ecolor='black', elinewidth=0.5, label='r0T')
	plt.plot(r0Array, r0Array, label='True r0')
	plt.legend()
	plt.xlabel("True r_0")
	plt.ylabel("Measured r_0")
	plt.savefig("Measured Relationship.pdf", bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(7, 5))
	plt.figure(1).add_axes((0, 0, 1, 0.8))
	r0LDiffArray = r0LArray-r0Array
	r0TDiffArray = r0TArray-r0Array
	plt.plot(r0Array, r0LDiffArray, label='r0L')
	plt.plot(r0Array, r0TDiffArray, label='r0T')
	plt.legend()
	plt.xlabel("True r_0")
	plt.ylabel("Measured r_0")
	plt.savefig("Measured Relationship Difference.pdf", bbox_inches='tight')
	plt.close()

	dictionary = {'r0': r0Array, 'r0L': r0LArray, 'r0LError':r0LErrorArray, 'r0T': r0TArray, 'r0TError': r0TErrorArray}
	writeData = pd.DataFrame(dictionary)
	time = datetime.now()
	Name = time.strftime("%H%M%S")
	writer = pd.ExcelWriter("r0 data.xlsx", mode='a')
	writeData.to_excel(writer, sheet_name=Name)


FindMeasuredR0()
