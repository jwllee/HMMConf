package nl.tue.alignment.algorithms.implementations;

import java.util.Arrays;

import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import nl.tue.alignment.Canceler;
import nl.tue.alignment.Utils;
import nl.tue.alignment.Utils.Statistic;
import nl.tue.alignment.algorithms.Queue;
import nl.tue.alignment.algorithms.ReplayAlgorithm;
import nl.tue.alignment.algorithms.VisitedSet;
import nl.tue.alignment.algorithms.datastructures.HashBackedPriorityQueue;
import nl.tue.alignment.algorithms.datastructures.SortedHashBackedPriorityQueue;
import nl.tue.alignment.algorithms.datastructures.VisitedHashSet;
import nl.tue.alignment.algorithms.syncproduct.SyncProduct;
import nl.tue.astar.util.ilp.LPMatrixException;

/**
 * The replay algorithm implements a replayer using the AStar skeleton. It
 * leaves implementations of the heuristic to implementing subclasses.
 * 
 * A few assumptions are made: <br />
 * 
 * Places can at most hold 3 tokens. If three tokens are in a place, the
 * preceding transitions are not enabled. If this situation is encountered, a
 * flag is set in the result Utils.ENABLINGBLOCKEDBYOUTPUT.
 * 
 * The cost function does not exceed 16777215. Since only 3 bytes are used to
 * keep it, the value of the cost function cannot exceed this limit. If this
 * happens, the algorithm ignores it, but sets the flag
 * Utils.COSTFUNCTIONOVERFLOW
 * 
 * The heuristic function does not exceed 16777214. Since only 3 bytes are used
 * to keep it and one value is reserved for infinity, the value of the heuristic
 * function cannot exceed this limit. If this happens, the algorithm sets the
 * flag Utils.HEURISTICFUNCTIONOVERFLOW and continues with the maximum heuristic
 * allowed.
 * 
 * Because of the maximum value of the cost function, it is wise not to set too
 * high values for the costs of firing certain transitions.
 */

/*- 
 * Internally, information is stored in two arrays. These arrays are double arrays, 
 * i.e. the used memory is split up into smaller blocks,
 * which allows for more efficient storage in memory, as well as synchronization on blocks
 * in future multi threaded versions.
 * 
 * markingLo and markingHi store the markings reached. Each marking is represented by 
 * two arrays of bytes (a low and high) array. Within these, each place has 1 bit.
 * i.e. [1000 1000][000.....] as low bytes and [1000 0000][010. ....] as high bytes 
 * together represent the marking with 3 tokens in place 0, 1 token in place 4 and 2 
 * tokens in place 9, assuming 11 places in the model (hence the 5 unused trailng bits). 
 * These bytes are serialized into markingLo and markingHi.
 * 
 * The array ptl_g stores the low bits of the preceding transition relation in the first byte and
 * the remaining 3 bytes are used to store the value for g (unsigned, i.e. assumed maximum 
 * is 16777215).
 * 
 * The array e_pth_h consists of three parts. The highest bit is a flag indicating if the 
 * stored heuristic is exact or estimated. Then, 7 bits are used to store the high bits 
 * of the preceding transition relation. The remaining 3 bytes are used to store the estimated 
 * value of h (Unsigned, assumed maximum is 16777214, since 16777215 stands for infinite).
 * 
 * The ptl relation is composed of the 8 low bits from ptl_g and the 7 high bits from pth_h. 
 * Combined with a highest bit equal to 0 they form a int value which is interpreted signed, 
 * i.e. maximum 32767.
 * 
 * The array c_p uses the highest bit to indicate that a marking is in the closed set. The
 * remaining 31 bits are used to store the predecessor index.
 * 
 * @author bfvdonge
 *
 */
abstract class AbstractReplayAlgorithm extends AbstractReplayAlgorithmDataStore {

	//	private static final int PTRANSLOMASK = 0b11111111000000000000000000000000;

	//	private static final int PTRANSHIMASK = 0b01111111000000000000000000000000;

	protected static final int RESTART = -1;

	/**
	 * Stores the closed set
	 */
	protected VisitedSet visited;

	/**
	 * The synchronous product under investigation
	 */
	protected final SyncProduct net;

	/**
	 * Stores the open set as a priority queue
	 */
	protected Queue queue;

	/**
	 * Indicate if moves should be considered totally ordered.
	 */
	protected boolean moveSorting;

	/**
	 * Stores the selected debug level
	 */
	protected final Debug debug;

	/**
	 * Flag indicating if exact solutions should be kept separate
	 */
	protected final boolean preferExact;

	protected int pollActions;
	protected int closedActions;
	protected int queueActions;
	protected int edgesTraversed;
	protected int markingsReached;
	protected int heuristicsComputed;
	protected int heuristicsEstimated;
	protected int heuristicsDerived;
	protected int alignmentLength;
	protected int alignmentCost;
	protected int setupTime;
	protected int runTime;
	protected int iteration;

	protected long startConstructor;
	protected int numPlaces;
	private boolean queueSorting;
	//	private boolean multiThreading;
	protected long timeoutAtTimeInMillisecond;
	private int maximumNumberOfStates;
	protected TObjectIntMap<Utils.Statistic> replayStatistics;

	public AbstractReplayAlgorithm(SyncProduct product, boolean moveSorting, boolean queueSorting,
			boolean preferExact) {
		this(product, moveSorting, queueSorting, preferExact, Debug.NONE);
	}

	public AbstractReplayAlgorithm(SyncProduct net, boolean moveSorting, boolean queueSorting, boolean preferExact,
			Debug debug) {
		this.queueSorting = queueSorting;
		this.preferExact = preferExact;
		//		this.multiThreading = multiThreading;
		this.debug = debug;
		startConstructor = System.nanoTime();

		this.numPlaces = net.numPlaces();
		this.net = net;
		this.moveSorting = moveSorting;

		// Array used internally for firing transitions.
		firingMarking = new byte[net.numPlaces()];
		hashCodeMarking = new byte[net.numPlaces()];
		equalMarking = new byte[net.numPlaces()];

		replayStatistics = new TObjectIntHashMap<>(20);

		this.setupTime = (int) ((System.nanoTime() - startConstructor) / 1000);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see nl.tue.alignment.algorithms.ReplayAlgorithmInterface#getStatistics()
	 */
	@Override
	public TObjectIntMap<Utils.Statistic> getStatistics() {
		return replayStatistics;
	}

	protected void fillStatistics(int[] alignment) {
		replayStatistics.put(Utils.Statistic.POLLACTIONS, pollActions);
		replayStatistics.put(Utils.Statistic.CLOSEDACTIONS, closedActions);
		replayStatistics.put(Utils.Statistic.QUEUEACTIONS, queueActions);
		replayStatistics.put(Utils.Statistic.EDGESTRAVERSED, edgesTraversed);
		replayStatistics.put(Utils.Statistic.MARKINGSREACHED, markingsReached);
		replayStatistics.put(Utils.Statistic.HEURISTICSCOMPUTED, heuristicsComputed);
		replayStatistics.put(Utils.Statistic.HEURISTICSDERIVED, heuristicsDerived);
		replayStatistics.put(Utils.Statistic.HEURISTICSESTIMATED, heuristicsEstimated);
		replayStatistics.put(Utils.Statistic.ALIGNMENTLENGTH, alignmentLength);
		replayStatistics.put(Utils.Statistic.COST, alignmentCost);
		replayStatistics.put(Utils.Statistic.EXITCODE, alignmentResult);
		replayStatistics.put(Utils.Statistic.RUNTIME, runTime);
		replayStatistics.put(Utils.Statistic.SETUPTIME, setupTime);
		replayStatistics.put(Utils.Statistic.TOTALTIME, setupTime + runTime);
		replayStatistics.put(Utils.Statistic.MAXQUEUELENGTH, queue.maxSize());
		replayStatistics.put(Utils.Statistic.MAXQUEUECAPACITY, queue.maxCapacity());
		replayStatistics.put(Utils.Statistic.VISITEDSETCAPACITY, visited.capacity());
		replayStatistics.put(Utils.Statistic.TRACELENGTH, net.numEvents());
		replayStatistics.put(Utils.Statistic.PLACES, net.numPlaces());
		replayStatistics.put(Utils.Statistic.TRANSITIONS, net.numTransitions());
		replayStatistics.put(Utils.Statistic.LMCOST,
				getCostForType(alignment, SyncProduct.LOG_MOVE, SyncProduct.LOG_MOVE));
		replayStatistics.put(Utils.Statistic.MMCOST,
				getCostForType(alignment, SyncProduct.MODEL_MOVE, SyncProduct.TAU_MOVE));
		replayStatistics.put(Utils.Statistic.SMCOST,
				getCostForType(alignment, SyncProduct.SYNC_MOVE, SyncProduct.SYNC_MOVE));
		replayStatistics.put(Utils.Statistic.MEMORYUSED, (int) (getEstimatedMemorySize() / 1024));
	}

	public long getEstimatedMemorySize() {

		// each array has 4 bytes overhead for storing the size
		long val = super.getEstimatedMemorySize();

		// count the capacity of the queue
		val += queue.getEstimatedMemorySize();
		// count the capacity of the visistedSet
		val += visited.getEstimatedMemorySize();

		return val;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * nl.tue.alignment.algorithms.ReplayAlgorithmInterface#run(nl.tue.alignment.
	 * Progress, int, int, int)
	 */
	@Override
	public int[] run(Canceler canceller, int timeoutMilliseconds, int maximumNumberOfStates, int costUpperLimit)
			throws LPMatrixException {
		if (maximumNumberOfStates <= 0) {
			this.maximumNumberOfStates = Integer.MAX_VALUE;
		} else {
			this.maximumNumberOfStates = maximumNumberOfStates;
		}
		pollActions = 0;
		closedActions = 0;
		queueActions = 0;
		edgesTraversed = 0;
		markingsReached = 0;
		heuristicsComputed = 0;
		heuristicsEstimated = 0;
		heuristicsDerived = 0;
		iteration = -1;

		return runReplayAlgorithm(canceller, System.nanoTime(),
				timeoutMilliseconds <= 0 ? Integer.MAX_VALUE : timeoutMilliseconds, costUpperLimit);
	}

	protected int[] runReplayAlgorithm(Canceler canceller, long startTime, int timeoutMilliseconds, int costUpperBound)
			throws LPMatrixException {

		//		int[] trans = new int[net.numTransitions()];
		//		for (int t = net.numTransitions(); t-- > 0;) {
		//			trans[t] = t;
		//		}
		//		Utils.shuffleArray(trans, new Random());
		timeoutAtTimeInMillisecond = System.currentTimeMillis() + timeoutMilliseconds;

		debug.println(Debug.DOT, "Digraph D {");
		restartLoop: do {
			int markingsReachedInRun = 1;
			int closedActionsInRun = 0;
			int[] alignment = null;
			int f_Score = -1;
			try {
				initializeIteration();
				debug.println(Debug.DOT, "subgraph cluster_" + iteration);
				debug.println(Debug.DOT, "{");
				debug.println(Debug.DOT, "label=<Iteration " + iteration + ">;");
				debug.println(Debug.DOT, "rankdir=LR;");
				debug.println(Debug.DOT, "color=black;");

				assert queue.size() == markingsReachedInRun - closedActionsInRun;

				byte[] marking_m = new byte[numPlaces];

				queueLoop: while (!queue.isEmpty() && (System.currentTimeMillis() < timeoutAtTimeInMillisecond)
						&& markingsReachedInRun < maximumNumberOfStates && !canceller.isCanceled()) {
					assert queue.size() == markingsReachedInRun - closedActionsInRun;

					int m = queue.peek();
					int bm = m >>> blockBit;
					int im = m & blockMask;

					f_Score = getFScore(bm, im);

					// check cost upper limit
					if (f_Score > costUpperBound) {
						break queueLoop;
					}

					switch (closeOrUpdateMarking(m, marking_m, bm, im)) {
						case FINALMARKINGFOUND :
							alignmentResult |= Utils.OPTIMALALIGNMENT;
							alignment = handleFinalMarkingReached(startTime, m);
							return alignment;
						case CLOSEDINFEASIBLE :
							closedActionsInRun++;
							//$FALL-THROUGH$
						case REQUEUED :
							continue queueLoop;
						case RESTARTNEEDED :
							continue restartLoop;
						//							return runReplayAlgorithm(startTime);
						case CLOSEDSUCCESSFUL :
							closedActionsInRun++;
					}
					markingsReachedInRun += expandMarking(m, marking_m, bm, im);
				} // end While
				alignmentResult &= ~Utils.OPTIMALALIGNMENT;
				alignmentResult |= Utils.FAILEDALIGNMENT;
				if (!queue.isEmpty()) {
					alignment = handleFinalMarkingReached(startTime, queue.peek());
				} else {
					alignment = getAlignmentWhenEmptyQueueReached(startTime);
					runTime = (int) ((System.nanoTime() - startTime) / 1000);
				}
				return alignment;
			} finally {
				if (System.currentTimeMillis() >= timeoutAtTimeInMillisecond) {
					alignmentResult |= Utils.TIMEOUTREACHED;
				}
				if (markingsReachedInRun >= maximumNumberOfStates) {
					alignmentResult |= Utils.STATELIMITREACHED;
				}
				if (f_Score > costUpperBound) {
					alignmentResult |= Utils.COSTLIMITREACHED;
				}
				if (canceller.isCanceled()) {
					alignmentResult |= Utils.CANCELLED;
				}
				if (alignmentResult == Utils.FAILEDALIGNMENT && queue.isEmpty()) {
					// no final marking found, no timeout, state limit, cost limit, or cancellation...
					// queue must be empty because final marking is unreachable.
					alignmentResult |= Utils.FINALMARKINGUNREACHABLE;
				}
				terminateIteration(alignment, markingsReachedInRun, closedActionsInRun);
			}
		} while ((alignmentResult & Utils.OPTIMALALIGNMENT) == 0 && //
				(alignmentResult & Utils.FAILEDALIGNMENT) == 0);
		return null;
	}

	protected int[] getAlignmentWhenEmptyQueueReached(long startTime) {
		return new int[0];
	}

	protected int expandMarking(int m, byte[] marking_m, int bm, int im) {
		int markingsReachedInExpand = 0;
		// iterate over all transitions
		for (int t = 0; t < net.numTransitions() && (System.currentTimeMillis() < timeoutAtTimeInMillisecond); t++) {
			//				for (int t = net.numTransitions(); t-- > 0;) {
			//				for (int t : trans) {

			// check for enabling
			if (isEnabled(marking_m, t, bm, im)) {
				edgesTraversed++;

				// t is allowed to fire.
				byte[] marking_n = fire(marking_m, t, bm, im);

				// check if n already reached before
				int newIndex = block * blockSize + indexInBlock;
				int n = visited.add(marking_n, newIndex);
				// adding the marking to the algorithm is handled by the VisitedSet.

				int bn = n >>> blockBit;
				int in = n & blockMask;
				//				getLockForComputingEstimate(bn, in);

				try {
					if (n == newIndex) {
						markingsReached++;
						markingsReachedInExpand++;
					}

					//					System.out.println("   Fire " + t + ": " + Utils.print(getMarking(n), net.numPlaces()));

					if (!isClosed(bn, in)) {
						// n is a fresh marking, not in the closed set
						// compute the F score on this path
						int tmpG = getGScore(bm, im) + net.getCost(t);

						if (tmpG < getGScore(bn, in)) {
							writeEdgeTraversed(this, m, t, n, "");

							// found a inter path to n.
							setGScore(bn, in, tmpG);

							// set predecessor
							setPredecessor(bn, in, m);
							setPredecessorTransition(bn, in, t);

							if (!hasExactHeuristic(bn, in)) {
								// estimate is not exact, so derive a new estimate (note that h cannot decrease here)
								deriveOrEstimateHValue(m, bm, im, t, n, bn, in);
							}

							// update position of n in the queue
							int s = queue.size();
							addToQueue(n);
							assert n == newIndex ? queue.size() == s + 1 : queue.size() == s;

						} else if (!hasExactHeuristic(bn, in)) {
							// not a new marking
							assert n < newIndex;
							//tmpG >= getGScore(n), i.e. we reached state n through a longer path.

							// G shore might not be an improvement, but see if we can derive the 
							// H score. 
							deriveOrEstimateHValue(m, bm, im, t, n, bn, in);

							if (hasExactHeuristic(bn, in)) {
								writeEdgeTraversed(this, m, t, n, ",color=gray19");
								// marking is now exact and was not before. 
								assert queue.contains(n);
								int s = queue.size();
								addToQueue(n);
								assert queue.size() == s;
							}
						} else {
							// not a new marking
							assert n < newIndex;
							// reached a marking of which F score is higher than current F score
							writeEdgeTraversed(this, m, t, n, ",style=dashed,color=gray19,arrowtail=tee");
						}
					} else {
						// reached an already closed marking
						writeEdgeTraversed(this, m, t, n, ",style=dashed,color=gray19,arrowtail=box");
					}
				} finally {
					//					releaseLockForComputingEstimate(bn, in);
				} // end Try processing n
			} // end If enabled
		} // end for transitions
		processedMarking(m, bm, im);
		return markingsReachedInExpand;
	}

	//	private transient TIntObjectMap<TIntSet> logMove2states = null;

	protected void writeEdgeTraversed(ReplayAlgorithm algorithm, int fromMarking, int transition, int toMarking,
			String extra) {
		if (debug == Debug.DOT) {
			if (transition >= 0) {
				extra = ",weight=" + (1000 - net.getTransitionPathLength(transition)) + extra;
			}
			//			if (logMove2states == null) {
			//				logMove2states = new TIntObjectHashMap<>(algorithm.getNet().numEvents());
			//			}
			// keep track of logMoves
			if (transition >= 0 && algorithm.getNet().getTypeOf(transition) == SyncProduct.LOG_MOVE) {
				//				int[] evts = algorithm.getNet().getEventOf(transition);
				//				logMove2states.putIfAbsent(evts[0], new TIntHashSet());
				//				logMove2states.get(evts[0]).add(fromMarking);
				//				if (evts.length > 1) {
				//					logMove2states.putIfAbsent(evts[evts.length - 1], new TIntHashSet());
				//					logMove2states.get(evts[evts.length - 1]).add(toMarking);
				//				}
				debug.println(debug,
						"{rank=same; i" + iteration + "m" + fromMarking + "; i" + iteration + "m" + toMarking + "  }");
			}
		}
		debug.writeEdgeTraversed(algorithm, fromMarking, transition, toMarking, extra);

	}

	protected CloseResult closeOrUpdateMarking(int m, byte[] marking_m, int bm, int im) {
		//				System.out.println("Main waiting for " + bm + "," + im);
		int heuristic;
		//		getLockForComputingEstimate(bm, im);
		//				System.out.println("Main locking " + bm + "," + im);
		try {
			if (m != queue.peek()) {
				// a parallel thread may have demoted m because the heuristic
				// changed from estimated to exact.
				return CloseResult.REQUEUED;
			}
			m = queue.poll();
			pollActions++;

			if (isFinal(m)) {
				assert queue.isEmpty() || getFScore(queue.peek()) >= getFScore(m);
				return CloseResult.FINALMARKINGFOUND;
			}

			fillMarking(marking_m, bm, im);
			if (!hasExactHeuristic(bm, im)) {

				// compute the exact heuristic
				heuristic = getExactHeuristic(m, marking_m, bm, im);
				if (heuristic == RESTART) {
					setClosed(bm, im);
					closedActions++;
					return CloseResult.RESTARTNEEDED;
				} else if (isInfinite(heuristic)) {
					// marking from which final marking is unreachable
					// ignore state and continue

					// set the score to exact score
					assert !queue.contains(m);
					setHScore(bm, im, heuristic, true);
					setClosed(bm, im);
					closedActions++;
					return CloseResult.CLOSEDINFEASIBLE;
				} else if (heuristic > getHScore(bm, im)) {
					// if the heuristic is higher push the head of the queue down
					// set the score to exact score
					assert !queue.contains(m);
					setHScore(bm, im, heuristic, true);
					addToQueue(m);
					return CloseResult.REQUEUED;
				} else {
					// continue with this marking
					// set the score to exact score
					setHScore(bm, im, heuristic, true);
				}
			}
			// add m to the closed set
			setClosed(bm, im);
			closedActions++;
			return CloseResult.CLOSEDSUCCESSFUL;
		} finally {
			// release the lock after potentially closing the marking
			//			releaseLockForComputingEstimate(bm, im);
			//					System.out.println("Main released " + bm + "," + im);
		}

	}

	/**
	 * @throws LPMatrixException
	 *             in subclasses using (I)LP
	 */
	protected void initializeIteration() throws LPMatrixException {
		initializeIterationInternal();
	}

	protected void initializeIterationInternal() {
		super.initializeIterationInternal();
		iteration++;

		this.visited = new VisitedHashSet(this, Utils.DEFAULTVISITEDSIZE);
		if (queueSorting) {
			if (preferExact) {
				this.queue = new SortedHashBackedPriorityQueue(this, Utils.DEFAULTQUEUESIZE, Integer.MAX_VALUE, true);
			} else {
				this.queue = new SortedHashBackedPriorityQueue(this, Utils.DEFAULTQUEUESIZE, Integer.MAX_VALUE, false);
			}
		} else {
			this.queue = new HashBackedPriorityQueue(this, Utils.DEFAULTQUEUESIZE, Integer.MAX_VALUE);
		}

		growArrays();

		alignmentLength = 0;
		alignmentCost = 0;
		alignmentResult = 0;
		runTime = 0;

		// get the initial marking
		byte[] initialMarking = net.getInitialMarking();
		// add to the set of markings
		int pos = addNewMarking(initialMarking);
		markingsReached++;

		int b = pos >>> blockBit;
		int i = pos & blockMask;
		// set predecessor to null
		setPredecessor(b, i, NOPREDECESSOR);
		setPredecessorTransition(b, i, 0);
		setGScore(b, i, 0);

		int heuristic;
		//			System.out.println("Main waiting for " + b + "," + i);
		//		getLockForComputingEstimate(b, i);
		//			System.out.println("Main locking " + b + "," + i);
		try {
			heuristic = getExactHeuristic(0, initialMarking, b, i);
			if (isInfinite(heuristic)) {
				alignmentResult |= Utils.FINALMARKINGUNREACHABLE;
				alignmentResult |= Utils.FAILEDALIGNMENT;
				return;
			}
			setHScore(b, i, heuristic, true);
		} finally {
			//			releaseLockForComputingEstimate(b, i);
			//				System.out.println("Main released " + b + "," + i);
		}

		addToQueue(0);
	}

	protected void addToQueue(int marking) {
		queue.add(marking);
		queueActions++;
	}

	protected void processedMarking(int marking, int blockMarking, int indexInBlock) {
		// skip. Can be used by subclasses to handle
	}

	protected int[] handleFinalMarkingReached(long startTime, int marking) {
		// Final marking reached.
		int n = getPredecessor(marking);
		int m2 = marking;
		int t;
		while (n != NOPREDECESSOR) {
			t = getPredecessorTransition(m2);

			writeEdgeTraversed(this, n, -1, m2, "color=red");
			alignmentLength++;
			alignmentCost += net.getCost(t);
			m2 = n;
			n = getPredecessor(n);
		}
		int[] alignment = new int[alignmentLength];
		n = getPredecessor(marking);
		m2 = marking;
		int l = alignmentLength;
		while (n != NOPREDECESSOR) {
			t = getPredecessorTransition(m2);
			alignment[--l] = t;
			m2 = n;
			n = getPredecessor(n);
		}

		runTime = (int) ((System.nanoTime() - startTime) / 1000);

		return alignment;
	}

	protected void writeEndOfAlignmentStats(int[] alignment, int markingsReachedInRun, int closedActionsInRun) {
		if (alignment != null) {
			debug.print(Debug.STATS, net.getLabel());
			for (Statistic s : Statistic.values()) {
				debug.print(Debug.STATS, Utils.SEP + replayStatistics.get(s));
			}
			debug.print(Debug.STATS, Utils.SEP + Runtime.getRuntime().maxMemory() / 1048576);
			debug.print(Debug.STATS, Utils.SEP + Runtime.getRuntime().totalMemory() / 1048576);
			debug.print(Debug.STATS, Utils.SEP + Runtime.getRuntime().freeMemory() / 1048576);
			debug.println(Debug.STATS);
		}
	}

	protected void writeEndOfAlignmentNormal(int[] alignment, int markingsReachedInRun, int closedActionsInRun) {
		if (alignment != null) {
			for (Statistic s : Statistic.values()) {
				debug.println(Debug.NORMAL, s + ": " + replayStatistics.get(s));
			}
		}
	}

	protected void writeEndOfAlignmentDot(int[] alignment, int markingsReachedInRun, int closedActionsInRun) {
		// close the graph

		//		if (debug == Debug.DOT) {
		//			final boolean[] done = new boolean[markingsReachedInRun];
		//			logMove2states.forEachEntry(new TIntObjectProcedure<TIntSet>() {
		//				public boolean execute(final int a, TIntSet b) {
		//					debug.print(debug, "subgraph cluster_event_");
		//					debug.println(debug, a + " {");
		//					debug.println(debug, "ordering=out;");
		//					debug.println(debug, "label=<Event " + a + ">;");
		//					b.forEach(new TIntProcedure() {
		//
		//						public boolean execute(int value) {
		//							debug.writeMarkingReached(AbstractReplayAlgorithm.this, value);
		//							done[value] = true;
		//							return true;
		//						}
		//					});
		//					debug.println(debug, "}");
		//					return true;
		//				}
		//			});
		//			debug.print(debug, "subgraph cluster_event_" + logMove2states.size() + " {");
		//			debug.println(debug, "label=<Event_" + logMove2states.size() + ">;");
		//			for (int m = 0; m < markingsReachedInRun; m++) {
		//				if (!done[m]) {
		//					debug.writeMarkingReached(this, m);
		//				}
		//			}
		//			debug.println(debug, "}");
		//		} else {
		for (int m = markingsReachedInRun; m-- > 0;) {
			debug.writeMarkingReached(this, m);
		}
		//		}
		// close the subgraph
		debug.println(Debug.DOT, "}");

		if (alignment != null) {

			// close the graph
			StringBuilder b = new StringBuilder();
			b.append("info [shape=plaintext,label=<");
			for (Statistic s : Statistic.values()) {
				b.append(s);
				b.append(": ");
				b.append(replayStatistics.get(s));
				b.append("<br/>");
			}
			b.append(">];");

			debug.println(Debug.DOT, b.toString());
			debug.println(Debug.DOT, "}");
		}
	}

	protected void terminateIteration(int[] alignment, int markingsReachedInRun, int closedActionsInRun) {
		// keep track of logMoves
		if (alignment != null) {
			fillStatistics(alignment);
		}
		if (debug == Debug.DOT) {
			writeEndOfAlignmentDot(alignment, markingsReachedInRun, closedActionsInRun);
		}
		if (debug == Debug.NORMAL) {
			writeEndOfAlignmentNormal(alignment, markingsReachedInRun, closedActionsInRun);
		}
		if (debug == Debug.STATS) {
			synchronized (debug.getOutputStream()) {
				writeEndOfAlignmentStats(alignment, markingsReachedInRun, closedActionsInRun);
			}
		}
		//		if (logMove2states != null) {
		//			logMove2states.clear();
		//			logMove2states = null;
		//		}
	}

	protected abstract void deriveOrEstimateHValue(int from, int fromBlock, int fromIndex, int transition, int to,
			int toBlock, int toIndex);

	protected abstract boolean isFinal(int marking);

	// Used internally in firing.
	private transient final byte[] firingMarking;

	protected byte[] fire(byte[] fromMarking, int transition, int block, int index) {
		// fire transition t in marking stored at block, index
		// First consumption:
		int[] input = net.getInput(transition);
		int[] output = net.getOutput(transition);
		System.arraycopy(fromMarking, 0, firingMarking, 0, numPlaces);

		for (int i = input.length; i-- > 0;) {
			firingMarking[input[i]]--;
		}
		for (int i = output.length; i-- > 0;) {
			firingMarking[output[i]]++;
		}
		//		
		//		for (int i = bm; i-- > 0;) {
		//			newMarking[i] = (byte) ((markingLo[block][bm * index + i] ^ input[i]) & 0xFF);
		//			byte tmp = (byte) ((newMarking[i] & input[i]) & 0xFF);
		//			newMarking[bm + i] = (byte) ((markingHi[block][bm * index + i] ^ tmp) & 0xFF);
		//		}
		//		// now production
		//		for (int i = bm; i-- > 0;) {
		//			byte tmp = (byte) ((newMarking[i] & output[i]) & 0xFF);
		//			newMarking[i] = (byte) ((newMarking[i] ^ output[i]) & 0xFF);
		//			newMarking[bm + i] = (byte) ((newMarking[bm + i] ^ tmp) & 0xFF);
		//		}

		return firingMarking;
	}

	protected void fire(byte[] marking, int transition) {
		// fire transition t in marking stored at block, index
		// First consumption:
		int[] input = net.getInput(transition);
		int[] output = net.getOutput(transition);

		for (int i = input.length; i-- > 0;) {
			marking[input[i]]--;
		}
		for (int i = output.length; i-- > 0;) {
			marking[output[i]]++;
		}
	}

	protected boolean isEnabled(byte[] marking, int transition, int block, int index) {
		// check enablement on the tokens and on the predecessor
		int preTransition = getPredecessorTransition(block, index);
		if (!moveSorting || preTransition <= transition || hasPlaceBetween(preTransition, transition)) {
			// allow firing only if there is a place between or if total order
			// is respected

			// copy marking
			System.arraycopy(marking, 0, firingMarking, 0, numPlaces);

			int[] input = net.getInput(transition);
			for (int i = input.length; i-- > 0;) {
				if (--firingMarking[input[i]] < 0) {
					return false;
				}
			}
			int[] output = net.getOutput(transition);
			for (int i = output.length; i-- > 0;) {
				if (++firingMarking[output[i]] > Byte.MAX_VALUE - 1
						|| /* additional overflow check */firingMarking[output[i]] < 0) {
					alignmentResult |= Utils.ENABLINGBLOCKEDBYOUTPUT;
					return false;
				}
			}

			//			for (int i = bm; i-- > 0;) {
			//				// ((markingLo OR markingHi) AND input) should be input.
			//				if (((markingLo[block][bm * index + i] | markingHi[block][bm * index + i]) & input[i]) != input[i]) {
			//					return false;
			//				}
			//			}
			//			// Firing semantics do not allow to produce more than 3 tokens
			//			// in a place ((markingLo AND markingHi) AND output) should be 0.
			//			byte[] output = net.getOutput(transition);
			//			for (int i = bm; i-- > 0;) {
			//				if (((markingLo[block][bm * index + i] & markingHi[block][bm * index + i]) & output[i]) != 0) {
			//					// if violated, signal in alignmentResult and continue
			//					alignmentResult |= Utils.ENABLINGBLOCKEDBYOUTPUT;
			//					return false;
			//				}
			//			}
			return true;
		} else {
			// not allowed to fire
			return false;
		}
	}

	/**
	 * returns true if there is a place common in the output set of transitionFrom
	 * and the input set of transitionTo
	 * 
	 * @param transitionFrom
	 * @param transitionTo
	 * @return
	 */
	public boolean hasPlaceBetween(int preTransition, int transition) {
		int[] input = net.getInput(transition);
		int[] output = net.getOutput(preTransition);
		// Note, input and output are SORTED LISTS.
		int i = 0, j = 0;
		while (i < input.length && j < output.length) {
			if (input[i] < output[j]) {
				i++;
			} else if (input[i] > output[j]) {
				j++;
			} else {
				assert input[i] == output[j];
				return true;
			}
		}
		return false;
	}

	private byte[] fillMarking(byte[] marking, int block, int index) {
		// add initial marking
		System.arraycopy(net.getInitialMarking(), 0, marking, 0, numPlaces);
		// fire all transitions in the sequence back
		int t;
		int m = getPredecessor(block, index);
		while (m != NOPREDECESSOR) {
			t = getPredecessorTransition(block, index);
			block = m >>> blockBit;
			index = m & blockMask;
			fire(marking, t);
			m = getPredecessor(block, index);
		}
		return marking;
	}

	public byte[] getMarking(int marking) {

		return fillMarking(new byte[numPlaces], marking >>> blockBit, marking & blockMask);
	}

	protected void fillMarking(byte[] markingArray, int marking) {
		fillMarking(markingArray, marking >>> blockBit, marking & blockMask);
	}

	public int addNewMarking(byte[] marking) {
		// allocate space for writing marking information in the block
		int pos = block * blockSize + indexInBlock;
		//		System.arraycopy(marking, 0, markingLo[block], bm * indexInBlock, bm);
		//		System.arraycopy(marking, bm, markingHi[block], bm * indexInBlock, bm);
		indexInBlock++;
		if (indexInBlock >= blockSize) {
			// write pointer moved over blockSize,
			// allocate a new block
			growArrays();
		}
		return pos;
	}

	protected void writeStatus() {
		debug.println(Debug.NORMAL, "Markings polled:   " + String.format("%,d", pollActions));
		debug.println(Debug.NORMAL, "   Markings reached:" + String.format("%,d", markingsReached));
		debug.println(Debug.NORMAL, "   Markings closed: " + String.format("%,d", closedActions));
		debug.println(Debug.NORMAL, "   FScore head:     " + getFScore(queue.peek()) + " = G: "
				+ getGScore(queue.peek()) + " + H: " + getHScore(queue.peek()));
		debug.println(Debug.NORMAL, "   Queue size:      " + String.format("%,d", queue.size()));
		debug.println(Debug.NORMAL, "   Queue actions:   " + String.format("%,d", queueActions));
		debug.println(Debug.NORMAL, "   Heuristics compu:" + String.format("%,d", heuristicsComputed));
		debug.println(Debug.NORMAL, "   Heuristics deriv:" + String.format("%,d", heuristicsDerived));
		debug.println(Debug.NORMAL, "   Heuristics est  :"
				+ String.format("%,d", (markingsReached - heuristicsComputed - heuristicsDerived)));
		debug.println(Debug.NORMAL, "   Estimated memory:" + String.format("%,d", getEstimatedMemorySize()));
		double time = (System.nanoTime() - startConstructor) / 1000000.0;
		debug.println(Debug.NORMAL, "   Time (ms):       " + String.format("%,f", time));
	}

	private transient final byte[] equalMarking;

	/**
	 * Checks equality of the stored marking1 to the given marking2.
	 * 
	 * @see SyncProduct.getInitialMarking();
	 * 
	 * @param marking1
	 * @param marking2
	 * @return
	 */
	public boolean equalMarking(int marking1, byte[] marking2) {
		//TODO: SMARTER!
		fillMarking(equalMarking, marking1);
		return Arrays.equals(equalMarking, marking2);
		//		int b = marking1 >>> blockBit;
		//		int i = marking1 & blockMask;
		//		return equalMarking(b, i, marking2);
	}

	//	protected boolean equalMarking(int block, int index, byte[] marking2) {
	//		
	//		for (int j = bm; j-- > 0;) {
	//			if (markingLo[block][bm * index + j] != marking2[j] || //
	//					markingHi[block][bm * index + j] != marking2[bm + j]) {
	//				return false;
	//			}
	//		}
	//		return true;
	//	}

	private transient final byte[] hashCodeMarking;

	/**
	 * Returns the hashCode of a stored marking
	 * 
	 * @param marking
	 * @return
	 */
	public int hashCode(int marking) {
		//TODO: SMARTER!
		fillMarking(hashCodeMarking, marking);
		return hashCode(hashCodeMarking);
	}

	/**
	 * Returns the hashCode of a stored marking which is provided as an array of
	 * length 2*bm, where the first bm bytes provide the low bits and the second bm
	 * bytes provide the high bits.
	 * 
	 * @see SyncProduct.getInitialMarking();
	 * @param marking
	 * @return
	 */
	public int hashCode(byte[] marking) {
		return Arrays.hashCode(marking);
	}

	public SyncProduct getNet() {
		return net;
	}

	private int getCostForType(int[] alignment, byte type1, byte type2) {
		int cost = 0;
		for (int i = 0; i < alignment.length; i++) {
			if (net.getTypeOf(alignment[i]) == type1 || net.getTypeOf(alignment[i]) == type2) {
				cost += net.getCost(alignment[i]);
			}
		}
		return cost;
	}

	public int getLastRankOf(int marking) {
		int m = marking;
		int trans;
		int evt = SyncProduct.NORANK;
		while (m > 0) {
			trans = getPredecessorTransition(m);
			evt = Math.max(evt, net.getRankOf(trans));
			m = getPredecessor(m);
		}
		return evt;
	}

	public void putStatistic(Statistic stat, int value) {
		replayStatistics.put(stat, value);
	}

	/**
	 * Returns the h score for the given marking
	 * 
	 * @param marking
	 * @return
	 */
	public abstract int getExactHeuristic(int marking, byte[] markingArray, int markingBlock, int markingIndex);

	public int getIterationNumber() {
		return iteration;
	}

	public int getPathLength(int marking) {
		if (marking == NOPREDECESSOR) {
			return 0;
		} else {
			return net.getTransitionPathLength(getPredecessorTransition(marking))
					+ getPathLength(getPredecessor(marking));
		}
	}

}
