package nl.tue.alignment.algorithms.implementations;

import java.util.Arrays;
import java.util.Random;

import gnu.trove.iterator.TIntIterator;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import lpsolve.LpSolve;
import lpsolve.LpSolveException;
import nl.tue.alignment.Utils;
import nl.tue.alignment.Utils.Statistic;
import nl.tue.alignment.algorithms.ReplayAlgorithm;
import nl.tue.alignment.algorithms.syncproduct.SyncProduct;
import nl.tue.astar.util.ilp.LPMatrixException;

/**
 * Implements a variant of AStar intest path algorithm for alignments. It uses
 * (I)LP to estimate the remaining distance.
 * 
 * This implementation can NOT be used for prefix alignments. The final marking
 * has to be reached as this is assumed in the underlying (I)LP.
 * 
 * @author bfvdonge
 * 
 */
public class AStarLargeLP extends AbstractLPBasedAlgorithm {

	/**
	 * Special version of AStarLargeLP that builds the entire search space, i.e.
	 * this version never terminates correctly, but the search simply stops when the
	 * queueis empty.
	 * 
	 * @author bfvdonge
	 *
	 */
	public static class Full extends AStarLargeLP {

		private int firstFinal = -1;

		public Full(SyncProduct product) {
			super(product);
		}

		public Full(SyncProduct product, boolean moveSorting, boolean queueSorting, Debug debug) {
			super(product, moveSorting, queueSorting, debug);
		}

		public Full(SyncProduct product, boolean moveSorting, boolean useInteger, Debug debug, int[] splitpoints) {
			super(product, moveSorting, useInteger, debug, splitpoints);
		}

		public Full(SyncProduct product, boolean moveSorting, boolean useInteger, int initialSplits, Debug debug) {
			super(product, moveSorting, useInteger, initialSplits, debug);
		}

		@Override
		protected boolean isFinal(int marking) {
			if (firstFinal < 0 && super.isFinal(marking)) {
				firstFinal = marking;
			}
			return false;
		}

		@Override
		protected int[] getAlignmentWhenEmptyQueueReached(long startTime) {
			if (this.firstFinal < 0) {
				return super.getAlignmentWhenEmptyQueueReached(startTime);
			} else {
				alignmentResult &= ~Utils.FAILEDALIGNMENT;
				alignmentResult |= Utils.OPTIMALALIGNMENT;
				return handleFinalMarkingReached(startTime, this.firstFinal);
			}
		}
	}

	private static Random random = new Random(41343);

	protected int heuristicsComputedInRun = 0;

	//	protected int numRows;
	//	protected int numCols;
	private int[] indexMap;

	private final TIntObjectMap<TIntList> rank2LSMove = new TIntObjectHashMap<>();

	private int numRanks;

	private int modelMoves;

	private SyncProduct product;
	private boolean[] useInteger;
	private int maxRankExact;
	private int maxRankMarking;

	public AStarLargeLP(SyncProduct product) {
		this(product, false, false, Debug.NONE);
	}

	/**
	 * 
	 * @param product
	 * @param moveSorting
	 * @param useInteger
	 * @param debug
	 * @param splitpoints
	 *            provides the initial splitpoints for this sync product. This is an
	 *            array of ranks of log-move transitions. If event at rank 2 is
	 *            problematic, the array should be [2]. In linear traces, the rank
	 *            is the index of the event in the trace.
	 */
	public AStarLargeLP(SyncProduct product, boolean moveSorting, boolean useInteger, Debug debug, int[] splitpoints) {
		this(product, moveSorting, useInteger, splitpoints.length + 1, false, debug);

		if (splitpoints.length > 0) {
			System.arraycopy(splitpoints, 0, this.splitpoints, 1, splitpoints.length);
		}
		this.splitpoints[this.splitpoints.length - 1] = (numRanks + 1);
		splits = splitpoints.length + 1;

		this.setupTime = (int) ((System.nanoTime() - startConstructor) / 1000);
	}

	/**
	 * 
	 * @param product
	 * @param moveSorting
	 * @param useInteger
	 * @param initialBins
	 * @param debug
	 */
	public AStarLargeLP(SyncProduct product, boolean moveSorting, boolean useInteger, Debug debug) {
		this(product, moveSorting, useInteger, 1, false, debug);
		this.splitpoints[1] = (numRanks + 1);

		this.setupTime = (int) ((System.nanoTime() - startConstructor) / 1000);
	}

	/**
	 * 
	 * @param product
	 * @param moveSorting
	 * @param useInteger
	 * @param initialBins
	 * @param debug
	 */
	public AStarLargeLP(SyncProduct product, boolean moveSorting, boolean useInteger, int initialSplits, Debug debug) {
		this(product, moveSorting, useInteger, initialSplits + 1, true, debug);

		this.setupTime = (int) ((System.nanoTime() - startConstructor) / 1000);
	}

	private AStarLargeLP(SyncProduct product, boolean moveSorting, boolean useInt, int initialBins, boolean initRandom,
			Debug debug) {
		super(product, moveSorting, true, true, debug);
		this.product = product;
		this.useInteger = new boolean[product.numTransitions()];
		Arrays.fill(this.useInteger, useInt);

		rank2LSMove.put(SyncProduct.NORANK, new TIntArrayList(10));
		numRanks = -1;
		TIntList set;
		for (int t = 0; t < product.numTransitions(); t++) {
			int r = product.getRankOf(t);
			set = rank2LSMove.get(r);
			if (set == null) {
				set = new TIntArrayList(3);
				rank2LSMove.put(r, set);
			}
			set.add(t);
			if (product.getRankOf(t) > numRanks) {
				numRanks = product.getRankOf(t);
			}
		}
		modelMoves = rank2LSMove.get(SyncProduct.NORANK).size();
		numRanks++;
		if (numRanks == 0) {
			// ensure model is not empty for empty trace
			numRanks = 1;
		}

		initialBins = Math.min(numRanks + 1, initialBins);
		if (initRandom) {
			splitpoints = new int[initialBins + 1];
			if (initialBins > 0) {
				double inc = (numRanks + 1) / (double) initialBins;
				double val = inc;
				for (int i = 1; i < splitpoints.length; i++) {
					splitpoints[i] = (int) (val + 0.5);
					val += inc;
				}
				// overwrite final value
				splitpoints[initialBins] = (numRanks + 1);
			}
		} else {
			splitpoints = new int[initialBins + 1];
		}
		splits = initialBins - 1;

		restarts = 0;
	}

	private int[] splitpoints;
	private int[] lastSplitpoints;
	private int[] move2col;

	private int rows;
	private int coefficients;

	private void init() throws LPMatrixException {
		lpSolutions.clear();
		lpSolutionsSize = 0;
		maxRankExact = SyncProduct.NORANK;
		maxRankMarking = 0;

		// only if intermediate splitpoint present, more rows are needed
		rows = 1 + (splitpoints.length - 1) * product.numPlaces();

		indexMap = new int[(splitpoints.length - 2) * modelMoves + product.numTransitions()];
		move2col = new int[(splitpoints.length - 1) * product.numTransitions()];
		Arrays.fill(move2col, -1);

		try {
			if (solver != null) {
				solver.deleteAndRemoveLp();
			}
			coefficients = 0;
			synchronized (LpSolve.class) {
				// reserve a row for randomsum to be minimized.
				solver = LpSolve.makeLp(rows, 0);
			}
			solver.setAddRowmode(false);

			double[] col = new double[1 + rows];
			int c = 0;

			int start = 1;
			for (int s = 1; s < splitpoints.length; s++) {

				// add model moves in this block (if any)
				// These are part of x_{s-1}
				c = addModelMovesToSolver(col, c, start, s - 1);

				//add log and sync moves in this block for all non-final ranks in the block.
				for (int e = splitpoints[s - 1]; e < splitpoints[s] - 1; e++) {
					// These are part of x_{s-1}
					c = addLogAndSyncMovesToSolver(col, c, start, s - 1, e, true);
				}
				//add log and sync moves in this block for final rank in the block, or full if last splitpoint

				// These is vector of y_{s}
				c = addLogAndSyncMovesToSolver(col, c, start, s - 1, (splitpoints[s] - 1), false);

				start += product.numPlaces();
			}

			// slack column
			Arrays.fill(col, 0);
			col[1] = -1;
			solver.addColumn(col);
			c++;
			coefficients++;
			// positive: do moves costs as late as possible
			// negative: do moves as early as possible
			solver.setObj(c, 1.0 / (c * 255));

			int r;
			// slack column equals sum other columns
			solver.setRh(1, 0);
			solver.setConstrType(1, LpSolve.EQ);

			// The first blocks have to result in a marking >= 0 after consumption
			for (r = 2; r <= rows - product.numPlaces(); r++) {
				solver.setConstrType(r, LpSolve.GE);
				solver.setRh(r, -product.getInitialMarking()[(r - 2) % product.numPlaces()]);
				coefficients++;
			}
			for (; r <= rows; r++) {
				solver.setConstrType(r, LpSolve.EQ);
				solver.setRh(r, product.getFinalMarking()[(r - 2) % product.numPlaces()]
						- product.getInitialMarking()[(r - 2) % product.numPlaces()]);
				coefficients++;
			}

			solver.setMinim();
			solver.setVerbose(0);

			solver.setScaling(LpSolve.SCALE_GEOMETRIC | LpSolve.SCALE_EQUILIBRATE | LpSolve.SCALE_INTEGERS);
			solver.setScalelimit(5);
			solver.setPivoting(LpSolve.PRICER_DEVEX | LpSolve.PRICE_ADAPTIVE);
			solver.setMaxpivot(250);
			solver.setBbFloorfirst(LpSolve.BRANCH_AUTOMATIC);
			solver.setBbRule(LpSolve.NODE_PSEUDONONINTSELECT | LpSolve.NODE_GREEDYMODE | LpSolve.NODE_DYNAMICMODE
					| LpSolve.NODE_RCOSTFIXING);
			solver.setBbDepthlimit(-50);
			solver.setAntiDegen(LpSolve.ANTIDEGEN_FIXEDVARS | LpSolve.ANTIDEGEN_STALLING);
			solver.setImprove(LpSolve.IMPROVE_DUALFEAS | LpSolve.IMPROVE_THETAGAP);
			solver.setBasiscrash(LpSolve.CRASH_NOTHING);
			solver.setSimplextype(LpSolve.SIMPLEX_DUAL_PRIMAL);

			//			try {
			//				solver.setBFPFromPath("bfp_etaPFI");
			//			} catch (Exception e) {
			//				// Gracefully ignore...
			//			}

			//			int res = solver.solve();
			//			double[] vars = new double[indexMap.length];
			//			solver.getVariables(vars);
			//			System.out.println(res + " : " + Arrays.toString(vars));
			//			debug.writeDebugInfo(Debug.NORMAL, "Solver: " + solver.getNrows() + " rows, " + solver.getNcolumns()
			//					+ " columns.");

		} catch (LpSolveException e) {
			solver.deleteAndRemoveLp();
			throw new LPMatrixException(e);
		}
		heuristicsComputedInRun = 0;
		varsMainThread = new double[indexMap.length + 1];
		tempForSettingSolution = new int[indexMap.length];

	}

	protected int addLogAndSyncMovesToSolver(double[] col, int c, int start, int currentSplitpoint, int rank,
			boolean full) throws LpSolveException {
		if (rank2LSMove.get(rank) != null) {
			int[] input;
			int[] output;
			TIntList list = rank2LSMove.get(rank);

			double n = 0;
			TIntIterator it = list.iterator();
			while (it.hasNext()) {
				int t = it.next();

				move2col[currentSplitpoint * net.numTransitions() + t] = c;

				//			for (int idx = 0; idx < list.size(); idx++) {
				//				int t = list.get(idx);

				Arrays.fill(col, 0);
				n += random.nextDouble();
				//				col[1] = splitpoints.length - currentSplitpoint - n / net.numTransitions();
				col[1] = currentSplitpoint + n / net.numTransitions();

				input = product.getInput(t);
				for (int i = 0; i < input.length; i++) {
					for (int p = 1 + start + input[i]; p < col.length; p += product.numPlaces()) {
						col[p] -= 1;
						coefficients++;
					}
				}

				output = product.getOutput(t);
				for (int i = 0; i < output.length; i++) {
					if (full) {
						for (int p = 1 + start + output[i]; p < col.length; p += product.numPlaces()) {
							col[p] += 1;
							coefficients++;
						}
					} else {
						for (int p = 1 + start + product.numPlaces() + output[i]; p < col.length; p += product
								.numPlaces()) {
							col[p] += 1;
							coefficients++;
						}
					}
				}
				solver.addColumn(col);
				indexMap[c] = t;
				c++;

				//				// SEMICONTINUOUS
				//				solver.setSemicont(c, true);
				//				// SEMICONTINUOUS LOWBO = 1
				//				solver.setLowbo(c, 1);
				solver.setLowbo(c, 0);

				coefficients++;
				solver.setUpbo(c, 1);
				coefficients++;
				solver.setInt(c, useInteger[t]);
				solver.setObj(c, product.getCost(t));
				coefficients++;
			} // for all sync/log moves
		} // if sync/logMoves
		return c;
	}

	protected int addModelMovesToSolver(double[] col, int c, int start, int currentSplitpoint) throws LpSolveException {
		int[] input;
		int[] output;
		if (rank2LSMove.get(SyncProduct.NORANK) != null) {
			TIntIterator it = rank2LSMove.get(SyncProduct.NORANK).iterator();
			// first the model moves in this block
			double n = 0;
			while (it.hasNext()) {
				Arrays.fill(col, 0);
				n += random.nextDouble();
				col[1] = currentSplitpoint + n / net.numTransitions();
				//				col[1] = splitpoints.length - currentSplitpoint - n / net.numTransitions();
				int t = it.next();

				move2col[currentSplitpoint * net.numTransitions() + t] = c;

				input = product.getInput(t);
				for (int i = 0; i < input.length; i++) {
					for (int p = 1 + start + input[i]; p < col.length; p += product.numPlaces()) {
						col[p] -= 1;
						coefficients++;
					}
				}
				output = product.getOutput(t);
				for (int i = 0; i < output.length; i++) {
					for (int p = 1 + start + output[i]; p < col.length; p += product.numPlaces()) {
						col[p] += 1;
						coefficients++;
					}
				}
				solver.addColumn(col);
				indexMap[c] = t;
				c++;
				//				// SEMICONTINUOUS
				//				solver.setSemicont(c, true);
				//				// SEMICONTINUOUS LOWBO = 1
				//				solver.setLowbo(c, 1);
				solver.setLowbo(c, 0);

				coefficients++;
				solver.setUpbo(c, 255);
				coefficients++;
				solver.setInt(c, useInteger[t]);
				solver.setObj(c, product.getCost(t));
				coefficients++;
			} // for all modelMoves
		} // if modelMoves
		return c;
	}

	@Override
	protected void growArrays() {
		super.growArrays();

	}

	@Override
	protected void initializeIteration() throws LPMatrixException {
		init();
		super.initializeIterationInternal();
	}

	protected double[] varsMainThread;
	protected int splits;
	protected int restarts;

	@Override
	public int getExactHeuristic(int marking, byte[] markingArray, int markingBlock, int markingIndex) {
		// find an available solver and block until one is available.

		int rank = (maxRankExact + 1);// marking == 0 ? SyncProduct.NORANK : getLastRankOf(marking);

		// the current path explains the events up to and including the event at maxRankExact with exact markings.
		// a state must exist with an estimated heuristic for the event maxRankExact+1. But the search cannot continue from there.
		// so, we separate maxRankExact+1 by putting the border at maxRankExact+2.
		// 

		int insert = 0;
		if (marking > 0) {
			// when stuck, force another rank to be added to the splitpoints. This ensures a new LpSolution.
			insert = Arrays.binarySearch(splitpoints, ++rank);
		}

		if (marking == 0 || insert >= 0 || rank > splitpoints[splitpoints.length - 1]) {
			// No event was explained yet, or the last explained event is already a splitpoint.
			// There's little we can do but continue with the replayer.
			//			debug.writeDebugInfo(Debug.NORMAL, "Solve call started");
			//			solver.printLp();

			int res = getExactHeuristic(solver, marking, markingArray, markingBlock, markingIndex, varsMainThread);
			//			debug.writeDebugInfo(Debug.NORMAL, "End solve: " + (System.currentTimeMillis() - s) / 1000.0 + " ms.");

			assert marking != 0 || (res >= 0 && res < HEURISTICINFINITE)
					|| (alignmentResult & Utils.TIMEOUTREACHED) == Utils.TIMEOUTREACHED;

			heuristicsComputedInRun++;

			int r = getLastRankOf(marking);
			if (r > maxRankExact) {
				maxRankExact = r;
				maxRankMarking = marking;
				//				System.out.println("Explained event at rank " + r + " exactly.");
			}

			return res;
		} else {
			//			System.out.print("Expanding splitpoints " + Arrays.toString(splitpoints));
			lastSplitpoints = splitpoints;
			insert = -insert - 1;
			splitpoints = Arrays.copyOf(splitpoints, splitpoints.length + 1);
			System.arraycopy(splitpoints, insert, splitpoints, insert + 1, splitpoints.length - insert - 1);
			splitpoints[insert] = rank;
			splits++;
			restarts++;
			debug.writeMarkingReached(this, marking);
			debug.writeMarkingReached(this, maxRankMarking, "peripheries=2");
			//			System.out.println(" to " + Arrays.toString(splitpoints));

			return RESTART;
			// Handle this case now.
		}

	}

	private int getExactHeuristic(LpSolve solver, int marking, byte[] markingArray, int markingBlock, int markingIndex,
			double[] vars) {
		long start = System.nanoTime();
		// start from correct right hand side
		try {

			int i = getSplitIndex(marking);
			//			i--;

			int r;
			for (r = 1; r <= i * product.numPlaces() + 1; r++) {
				solver.setRh(r, 0);
			}
			for (; r <= rows - product.numPlaces(); r++) {
				solver.setRh(r, -markingArray[(r - 2) % product.numPlaces()]);
			}
			for (; r <= rows; r++) {
				solver.setRh(r, product.getFinalMarking()[(r - 2) % product.numPlaces()]
						- markingArray[(r - 2) % product.numPlaces()]);
			}

			solver.defaultBasis();
			// set timeout in seconds;
			long remainingTime = timeoutAtTimeInMillisecond - System.currentTimeMillis();
			// round the remaining time up to the nearest second.
			solver.setTimeout(Math.max(1000, remainingTime + 999) / 1000);
			int solverResult = solver.solve();
			heuristicsComputed++;

			//			if (solverResult == LpSolve.INFEASIBLE || solverResult == LpSolve.NUMFAILURE) {
			//				// BVD: LpSolve has the tendency to give false infeasible or numfailure answers. 
			//				// It's unclear when or why this happens, but just in case...
			//				solverResult = solver.solve();
			//
			//			}
			if (solverResult == LpSolve.OPTIMAL) {
				// retrieve the solution
				solver.getVariables(vars);

				setNewLpSolution(marking, vars);

				// compute cost estimate
				double c = computeCostForVars(vars);

				if (c >= HEURISTICINFINITE) {
					alignmentResult |= Utils.HEURISTICFUNCTIONOVERFLOW;

					// continue with maximum heuristic value not equal to infinity.
					return HEURISTICINFINITE - 1;
				}

				// assume precision 1E-7 and round down
				return (int) (c + 1E-7);
			} else if (solverResult == LpSolve.INFEASIBLE) {
				return HEURISTICINFINITE;
			} else if (solverResult == LpSolve.TIMEOUT) {
				//				System.out.println("Remaining was: "+remainingTime);
				remainingTime = timeoutAtTimeInMillisecond - System.currentTimeMillis();
				//				System.out.println("Remaining is:  "+remainingTime);
				assert remainingTime <= 0;
				alignmentResult |= Utils.TIMEOUTREACHED;

				return HEURISTICINFINITE;
			} else {
				//					lp.writeLp("D:/temp/alignment/debugLP-Alignment.lp");
				System.err.println("Error code from LpSolve solver:" + solverResult);
				return HEURISTICINFINITE;
			}

		} catch (LpSolveException e) {
			e.printStackTrace();
			return HEURISTICINFINITE;
		} finally {
			solveTime += System.nanoTime() - start;

		}

	}

	@Override
	protected void setNewLpSolution(int marking, double[] solutionDouble) {
		// copy the solution from double array to byte array (rounding down)
		// and compute the maximum.
		Arrays.fill(tempForSettingSolution, 0);
		byte bits = 0;
		for (int i = tempForSettingSolution.length; i-- > 0;) {
			tempForSettingSolution[i] = ((int) (solutionDouble[i] + 1E-7));
			if (tempForSettingSolution[i] < ((int) (solutionDouble[i] - 1E-7))) {
				//rounded down
				useInteger[indexMap[i]] = true;
			}
			if (tempForSettingSolution[i] > (1 << bits)) {
				bits++;
			}
		}
		bits++;
		setNewLpSolution(marking, bits, tempForSettingSolution);
	}

	protected double computeCostForVars(double[] vars) {
		double c = 0;
		for (int t = vars.length - 1; t-- > 0;) {
			c += vars[t] * net.getCost(indexMap[t]);
		}
		return c;
	}

	//	@Override
	//	protected void setNewLpSolution(int marking, double[] solutionDouble) {
	//		// copy the solution from double array to byte array (rounding down)
	//		// and compute the maximum.
	//		Arrays.fill(tempForSettingSolution, 0);
	//		byte bits = 1;
	//		for (int i = solutionDouble.length; i-- > 0;) {
	//			tempForSettingSolution[indexMap[i]] += ((int) (solutionDouble[i] + 1E-7));
	//			if (tempForSettingSolution[indexMap[i]] > (1 << (bits - 1))) {
	//				bits++;
	//			}
	//		}
	//		setNewLpSolution(marking, bits, tempForSettingSolution);
	//	}

	protected void deriveOrEstimateHValue(int from, int fromBlock, int fromIndex, int transition, int to, int toBlock,
			int toIndex) {
		int splitIndex = getSplitIndex(from);

		int var = move2col[splitIndex * net.numTransitions() + transition];

		if (hasExactHeuristic(fromBlock, fromIndex) && var >= 0 && getHScore(fromBlock, fromIndex) != HEURISTICINFINITE
				&& (getLpSolution(from, var) >= 1)) {
			// from Marking has exact heuristic
			// we can derive an exact heuristic from it

			setDerivedLpSolution(from, to, var);
			// set the exact h score
			setHScore(toBlock, toIndex, getHScore(fromBlock, fromIndex) - net.getCost(transition), true);
			heuristicsDerived++;

			int r = getLastRankOf(to);
			if (r > maxRankExact) {
				maxRankExact = r;
				maxRankMarking = to;
				//				System.out.println("Explained event at rank " + r + " exactly.");
			}
		} else if (hasExactHeuristic(fromBlock, fromIndex) && getHScore(fromBlock, fromIndex) == HEURISTICINFINITE) {
			// marking from which final state cannot be reached
			setHScore(toBlock, toIndex, HEURISTICINFINITE, true);
			heuristicsDerived++;
		} else {
			if (isFinal(to)) {
				setHScore(toBlock, toIndex, 0, true);
				int r = getLastRankOf(to);
				if (r > maxRankExact) {
					maxRankExact = r;
					maxRankMarking = to;
					//				System.out.println("Explained event at rank " + r + " exactly.");
				}
			} else {
				int h = getHScore(fromBlock, fromIndex) - net.getCost(transition);
				if (h < 0) {
					h = 0;
				}
				if (h > getHScore(toBlock, toIndex)) {
					// estimated heuristic should not decrease.
					setHScore(toBlock, toIndex, h, false);
					heuristicsEstimated++;
				}
			}
		}

	}

	/**
	 * Returns the first index i in splitpoints such that splitpoints[i] >=
	 * getLastRankof(marking)
	 */
	private int getSplitIndex(int marking) {
		int e = getLastRankOf(marking) + 1;
		int i = 1;
		while (splitpoints[i] <= e) {
			i++;
		}
		return --i;
	}

	@Override
	public long getEstimatedMemorySize() {
		long val = super.getEstimatedMemorySize();
		// approximate memory for LpSolve
		val += 8 * coefficients * 2;
		return val;
	}

	@Override
	protected void fillStatistics(int[] alignment) {
		super.fillStatistics(alignment);
		putStatistic(Statistic.HEURISTICTIME, (int) (solveTime / 1000));
		putStatistic(Statistic.SPLITS, splits);
		putStatistic(Statistic.RESTARTS, restarts);
	}

	@Override
	protected void writeEndOfAlignmentDot(int[] alignment, int markingsReachedInRun, int closedActionsInRun) {
		for (int m = 0; m < markingsReachedInRun; m++) {
			if (!isClosed(m)) {
				if (isDerivedLpSolution(m)) {
					debug.writeMarkingReached(this, m, "color=blue,style=bold");
				} else if (hasExactHeuristic(m)) {
					debug.writeMarkingReached(this, m, "style=bold");
				} else {
					debug.writeMarkingReached(this, m, "style=dashed");
				}
			}
		}
		if (alignment != null) {
			lastSplitpoints = splitpoints;
		}
		// close the subgraph
		StringBuilder b = new StringBuilder();
		b.append("info" + iteration + " [shape=plaintext,label=<");
		b.append("Iteration: " + iteration);
		b.append("<br/>");
		b.append("Markings reached: " + markingsReachedInRun);
		b.append("<br/>");
		b.append("Markings closed: " + closedActionsInRun);
		b.append("<br/>");
		b.append("Heuristics computed: " + heuristicsComputedInRun);
		b.append("<br/>");
		b.append("Splitpoints: ");
		b.append(Arrays.toString(lastSplitpoints));
		b.append(">];");
		debug.println(Debug.DOT, b.toString());
		// close the subgraph
		debug.println(Debug.DOT, "}");
		if (alignment != null) {
			b = new StringBuilder();
			b.append("subgraph cluster_info {");
			b.append("label=<Global results>;");
			b.append("info [shape=plaintext,label=<");
			for (Statistic s : Statistic.values()) {
				b.append(s);
				b.append(": ");
				b.append(replayStatistics.get(s));
				b.append("<br/>");
			}
			b.append(">];");
			debug.println(Debug.DOT, b.toString());
			// close the subgraph
			debug.println(Debug.DOT, "}");
			// close the graph
			debug.println(Debug.DOT, "}");
		}
	}

	@Override
	protected void processedMarking(int marking, int blockMarking, int indexInBlock) {
		super.processedMarking(marking, blockMarking, indexInBlock);
		if (isDerivedLpSolution(marking)) {
			debug.writeMarkingReached(this, marking, "color=blue,style=bold");
		} else {
			debug.writeMarkingReached(this, marking, "style=bold");
		}
		byte[] removed = lpSolutions.remove(marking);
		lpSolutionsSize -= 12 + 4 + (removed != null ? removed.length : 0); // object size
		lpSolutionsSize -= 1 + 4 + 8; // used flag + key + value pointer
	}

	@Override
	protected void writeEndOfAlignmentStats(int[] alignment, int markingsReachedInRun, int closedActionsInRun) {
		if (alignment != null) {
			debug.print(Debug.STATS, net.getLabel());
			for (Statistic s : Statistic.values()) {
				debug.print(Debug.STATS, Utils.SEP + replayStatistics.get(s));
			}
			debug.print(Debug.STATS, Utils.SEP + Runtime.getRuntime().maxMemory() / 1048576);
			debug.print(Debug.STATS, Utils.SEP + Runtime.getRuntime().totalMemory() / 1048576);
			debug.print(Debug.STATS, Utils.SEP + Runtime.getRuntime().freeMemory() / 1048576);
			debug.print(Debug.STATS, Utils.SEP + toString(splitpoints));
			debug.println(Debug.STATS);
			debug.getOutputStream().flush();
		}
	}

	/**
	 * Returns a string representation of the contents of the specified array. The
	 * string representation consists of a list of the array's elements, enclosed in
	 * square brackets (<tt>"[]"</tt>). Adjacent elements are separated by the
	 * characters <tt>", "</tt> (a comma followed by a space). Elements are
	 * converted to strings as by <tt>String.valueOf(int)</tt>. Returns
	 * <tt>"null"</tt> if <tt>a</tt> is <tt>null</tt>.
	 *
	 * @param a
	 *            the array whose string representation to return
	 * @return a string representation of <tt>a</tt>
	 * @since 1.5
	 */
	private static String toString(int[] a) {
		if (a == null)
			return "null";
		int iMax = a.length - 1;
		if (iMax == -1)
			return "[]";

		StringBuilder b = new StringBuilder();
		b.append('[');
		for (int i = 0;; i++) {
			b.append(a[i]);
			if (i == iMax)
				return b.append(']').toString();
			b.append(" ");
		}
	}

	private String toString(double[] vars) {

		int[] tempForString = new int[net.numTransitions()];
		// copy the solution from double array to byte array (rounding down)
		// and compute the maximum.
		Arrays.fill(tempForSettingSolution, 0);
		byte bits = 1;
		for (int i = vars.length; i-- > 0;) {
			tempForString[indexMap[i]] += ((int) (vars[i] + 1E-7));
			if (tempForString[indexMap[i]] > (1 << (bits - 1))) {
				bits++;
			}
		}

		int iMax = tempForString.length - 1;
		if (iMax == -1)
			return "";

		StringBuilder b = new StringBuilder();
		for (int i = 0;; i++) {
			if (tempForString[i] > 0) {
				b.append(tempForString[i]);
				b.append(" ");
				b.append(net.getTransitionLabel(i));
			}
			if (i == iMax)
				return b.toString();
			if (tempForString[i] > 0) {
				b.append(", ");
			}
		}
	}

	private String toStringPerBlock(double[] vars) {

		int iMax = move2col.length - 1;
		if (iMax == -1)
			return "";
		StringBuilder b = new StringBuilder();
		for (int i = 0;; i++) {
			if (move2col[i] >= 0) {
				if (vars[move2col[i]] > 0) {
					b.append(((int) (vars[move2col[i]] + 1E-7)));
					b.append(" ");
					b.append(net.getTransitionLabel(indexMap[move2col[i]]));
				}
				if (i == iMax)
					return b.toString();
				if (i > 0 && i % net.numTransitions() == 0) {
					b.append(" || ");
				} else if (vars[move2col[i]] > 0) {
					b.append(", ");
				}
			}
		}
	}

	private double[] buildLpSolution(int marking) {
		double[] vars = new double[tempForSettingSolution.length];
		for (int v = 0; v < vars.length; v++) {
			vars[v] = getLpSolution(marking, v);
		}
		return vars;

	}

	@Override
	protected void writeEdgeTraversed(ReplayAlgorithm algorithm, int fromMarking, int transition, int toMarking,
			String extra) {
		if (debug == Debug.DOT && transition >= 0
				&& Arrays.binarySearch(splitpoints, algorithm.getNet().getRankOf(transition) + 1) >= 0) {
			debug.writeEdgeTraversed(algorithm, fromMarking, transition, toMarking,
					extra + ",style=\"tapered\",penwidth=\"5\"");
		} else {
			debug.writeEdgeTraversed(algorithm, fromMarking, transition, toMarking, extra);
		}
	}

}