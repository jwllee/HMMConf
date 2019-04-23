package nl.tue.alignment.algorithms.datastructures;

import nl.tue.alignment.algorithms.ReplayAlgorithm;

public class SortedHashBackedPriorityQueue extends HashBackedPriorityQueue {

	private final boolean preferExact;

	/**
	 * Creates a {@code HashBackedPriorityQueue} with the specified initial capacity
	 * that orders its elements according to the specified comparator.
	 * 
	 * @param algorithm
	 *            the algorithm the queue is used in
	 * @param initialCapacity
	 *            the initial capacity for this priority queue
	 * @param maxCost
	 *            the maximum cost for anything to be added to this queue
	 * 
	 * @throws IllegalArgumentException
	 *             if {@code initialCapacity} is less than 1
	 */

	public SortedHashBackedPriorityQueue(ReplayAlgorithm algorithm, int initialCapacity, int maxCost,
			boolean preferExact) {
		super(algorithm, initialCapacity, maxCost);
		this.preferExact = preferExact;
	}

	public SortedHashBackedPriorityQueue(ReplayAlgorithm algorithm, int initialCapacity, boolean preferExact) {
		super(algorithm, initialCapacity);
		this.preferExact = preferExact;
	}

	/**
	 * First order sorting is based on F score. Second order sorting on G score,
	 * where higher G score is better.
	 */
	@Override
	public boolean isBetter(int marking1, int marking2) {
		// retrieve stored cost
		int c1 = algorithm.getFScore(marking1);
		int c2 = algorithm.getFScore(marking2);

		// first order criterion: total F score.
		if (c1 != c2) {
			return c1 < c2; //
		}

		// second order sorting on exactness
		if (preferExact) {
			boolean b1 = algorithm.hasExactHeuristic(marking1);
			boolean b2 = algorithm.hasExactHeuristic(marking2);
			if (b1 != b2) {
				// if marking 1 has an exact heuristic and marking 2 does not, prefer marking 1.
				return b1;
			}
		}

		// Prefer lower explained rank
		// when both markings have exact or inexact heuristics, schedule the largest event number
		c1 = algorithm.getLastRankOf(marking1);
		c2 = algorithm.getLastRankOf(marking2);
		if (c1 != c2) {
			// more events explained;
			// if marking 1 explains more events (reaches a higher rank) prefer marking 1 over marking 2.
			return c1 > c2;
		}

		// Prefer higher G score
		// when both markings have exact or inexact heuristics, schedule the largest g score first
		if (algorithm.getGScore(marking1) > algorithm.getGScore(marking2)) {
			return true;
		} else if (algorithm.getGScore(marking1) < algorithm.getGScore(marking2)) {
			return false;
		}

		c1 = algorithm.getPathLength(marking1);
		c2 = algorithm.getPathLength(marking2);
		if (c1 != c2) {
			// prefer the longest path. (counter intuitive, but this ensures maximal sequentialization
			// of the LP solution already achieved.
			return c1 > c2;
		}

		// prefer the first marking reached
		return marking2 < marking1;
	}

	//	private int getLastEventNumber(int marking) {
	//		int eventNumber = -1;
	//		while (marking > 0 && eventNumber == -1) {
	//			eventNumber = algorithm.getNet().getEventOf(algorithm.getPredecessorTransition(marking));
	//			marking = algorithm.getPredecessor(marking);
	//		}
	//		return eventNumber;
	//	}
}
