#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
#elif PYQT_VER == 'PYQT4':
	#from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario
		self.ncities = len(self._scenario.getCities())

	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0):
		route = []
		cities = self._scenario.getCities()
		results = {}
		heap = []
		foundTour = False
		city = 0
		start_time = time.time()

		while city < self.ncities and time.time() - start_time < time_allowance:
			notFound = True
			route.append(city)
			while notFound:
				currentCity = route[-1]
				leastCity = None
				pathCost = np.inf
				for i in range(self.ncities):
					if cities[currentCity].costTo(cities[i]) < pathCost and i not in route:
						leastCity = i
						pathCost = cities[currentCity].costTo(cities[i])
				if leastCity is not None:
					route.append(leastCity)
				else:
					notFound = False
			if len(route) == self.ncities:
				sol = TSPSolution(self.getCities(route))
				heapq.heappush(heap, sol)
				foundTour = True
			city += 1
			route = []

		end_time = time.time()
		bssf = heapq.heappop(heap)

		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = None
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound( self, time_allowance=60.0 ):
		bssf = self.defaultRandomTour()["soln"]
		matrix = self.makeMatrix()
		self.lower_bound = 0
		self.state_count = 0
		count = 0

		#Generate an initial state
		mockState = State(self.state_count, matrix, self.lower_bound, [])
		ogState = self.reduceMatrix(mockState, self.state_count)
		self.state_count += 1

		#Start the queue and put the initial state into it
		heap = []
		heapq.heappush(heap, ogState)
		start_time = time.time()
		results = {}

		#This will now continue till I find the tour that is better then the BSSF or the time is up
		self.pruned = 0
		self.heapSize = 0
		while time.time() - start_time < time_allowance and len(heap) != 0:
			#Checks the largest the heap gets
			if len(heap) > self.heapSize:
				self.heapSize = len(heap)

			popState = heapq.heappop(heap)
			#First prunning
			if popState.lowerBound >= bssf.cost:
				self.pruned += 1
				continue

			#Now I start looping through and generating children for the popped state
			if popState.lowerBound < bssf.cost and len(popState.path) == self.ncities:
				bssf = TSPSolution(self.getCities(popState.path))
				count += 1
			for i in range(self.ncities):
				if i in popState.path:
					continue
				childState = self.generateState(popState, i)

				#Second Prunning
				if childState.lowerBound < bssf.cost:
					heapq.heappush(heap, childState)
				else:
					self.pruned += 1

		end_time = time.time()

		#Preps the return data
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = self.heapSize
		results['total'] = self.state_count
		results['pruned'] = self.pruned

		return results

	#Makes a matrix based on the data in the scenario passed into the branch and bound function
	def makeMatrix(self):
		cities = self._scenario.getCities()
		arr = []
		for i in range(self.ncities):
			row = []
			for j in range(self.ncities):
				dist = cities[i].costTo(cities[j])
				row.append(dist)
			arr.append(row)
		return np.array(arr)

	#Reduces the matrix present in whatever state is passed in
	def reduceMatrix(self, state, nextStateNum):
		lowerBound = state.lowerBound
		matrix = state.mtx.copy()
		path = state.path.copy()
		#Go through rows first then columns
		for i in range(2):
			for j in range(self.ncities):
				if i == 0:
					toEdit = matrix[j]
				else:
					toEdit = matrix[:,j]
				eMin = min(toEdit)
				if eMin == np.inf or eMin == 0:
					continue
				edited = toEdit - eMin
				lowerBound += eMin
				if i == 0:
					matrix[j] = edited
				else:
					matrix[:,j] = edited
		if self.lower_bound == 0:
			self.lower_bound = lowerBound
			stateCount = self.state_count
		else:
			stateCount = nextStateNum
		path.append(stateCount)
		state = State(stateCount, matrix, lowerBound, path)
		return state


	#This generates the children state based on the parent state
	def generateState(self, state, nextStateNum):
		matrix = state.mtx.copy()
		lowerBound = state.lowerBound

		#Here I will add the cost of new edge
		lowerBound += matrix[state.path[-1], nextStateNum]
		matrix[nextStateNum, state.path[-1]] = np.inf

		#Here I set up the infinity rows, columns and edge
		infArray = np.full(self.ncities, np.inf)
		matrix[state.path[-1]] = infArray
		matrix[:, nextStateNum] = infArray
		childState = State(nextStateNum, matrix, lowerBound, state.path)
		childState = self.reduceMatrix(childState, nextStateNum)
		self.state_count += 1


		return childState

	#Generates the route for the bssf
	def getCities(self, path):
		route = []
		for i in path:
			route.append(self._scenario.getCities()[i])
		return route

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy(self, time_allowance=60.0):
		pass

#This is a State class that holds all the info necessary to progress the algorithm.
class State:
	def __init__(self, stateNum, mtx, lowerBound, path):
		self.stateNum = stateNum
		self.mtx = mtx
		self.lowerBound = lowerBound
		self.path = path

	def __lt__(self, other):
		return self.lowerBound/len(self.path) < other.lowerBound/len(other.path)