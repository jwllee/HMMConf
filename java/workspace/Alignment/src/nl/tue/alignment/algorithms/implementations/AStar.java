package nl.tue.alignment.algorithms.implementations;

import lpsolve.LpSolve;
import lpsolve.LpSolveException;
import nl.tue.alignment.Utils;
import nl.tue.alignment.Utils.Statistic;
import nl.tue.alignment.algorithms.syncproduct.SyncProduct;
import nl.tue.astar.util.ilp.LPMatrix;
import nl.tue.astar.util.ilp.LPMatrix.SPARSE.LPSOLVE;
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
public class AStar extends AbstractLPBasedAlgorithm {

	/**
	 * Special version of AStar that builds the entire search space, i.e. this
	 * version never terminates correctly, but the search simply stops when the
	 * queueis empty.
	 * 
	 * @author bfvdonge
	 *
	 */
	public static class Full extends AStar {

		private int firstFinal = -1;

		public Full(SyncProduct product) throws LPMatrixException {
			super(product);
		}

		public Full(SyncProduct product, boolean moveSorting, boolean queueSorting, boolean preferExact,
				boolean isInteger, Debug debug) throws LPMatrixException {
			super(product, moveSorting, queueSorting, preferExact, isInteger, debug);
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

	protected final double[] rhf;

	private LPSOLVE matrix;

	public AStar(SyncProduct product) throws LPMatrixException {
		this(product, true, true, true, false, Debug.NONE);
	}

	public AStar(SyncProduct product, boolean moveSorting, boolean queueSorting, boolean preferExact, boolean isInteger,
			Debug debug) throws LPMatrixException {
		super(product, moveSorting, queueSorting, preferExact, debug);
		//		this.numberOfThreads = numberOfThreads;
		//		this.numRows = net.numPlaces();
		matrix = new LPMatrix.SPARSE.LPSOLVE(net.numPlaces(), net.numTransitions());

		// Set the objective to follow the cost function
		for (int t = net.numTransitions(); t-- > 0;) {
			matrix.setObjective(t, net.getCost(t));

			int[] input = net.getInput(t);
			for (int i = input.length; i-- > 0;) {
				// transition t consumes from place p, hence  incidence matrix
				// is -1;
				matrix.adjustMat(input[i], t, -1);
				assert matrix.getMat(input[i], t) <= 1;
			}
			int[] output = net.getOutput(t);
			for (int i = output.length; i-- > 0;) {
				matrix.adjustMat(output[i], t, 1);
				assert matrix.getMat(output[i], t) <= 1;
			}

			// Use integer variables if specified
			matrix.setInt(t, isInteger);
			// Set lower bound of 0
			matrix.setLowbo(t, 0);
			// Set upper bound to 255 (we assume unsigned byte is sufficient to store result)
			matrix.setUpbo(t, 255);

			if (debug != Debug.NONE) {
				matrix.setColName(t, net.getTransitionLabel(t));
			}

		}

		rhf = new double[net.numPlaces()];
		byte[] marking = net.getFinalMarking();
		for (int p = net.numPlaces(); p-- > 0;) {
			if (debug != Debug.NONE) {
				matrix.setRowName(p, net.getPlaceLabel(p));
			}
			// set the constraint to equality
			matrix.setConstrType(p, LPMatrix.EQ);
			rhf[p] = marking[p];
		}
		matrix.setMinim();

		this.setupTime = (int) ((System.nanoTime() - startConstructor) / 1000);
	}

	@Override
	protected void initializeIteration() throws LPMatrixException {
		//		try {
		solver = matrix.toSolver();

		//		try {
		//			solver.setBFPFromPath("bfp_etaPFI");
		//		} catch (Exception e) {
		//			// Gracefully ignore...
		//		}

		// bytes for solver
		bytesUsed = matrix.bytesUsed();
		varsMainThread = new double[net.numTransitions()];
		super.initializeIterationInternal();
	}

	@Override
	protected void growArrays() {
		super.growArrays();
	}

	protected double[] varsMainThread;

	@Override
	public int getExactHeuristic(int marking, byte[] markingArray, int markingBlock, int markingIndex) {
		// find an available solver and block until one is available.

		return getExactHeuristic(solver, marking, markingArray, markingBlock, markingIndex, varsMainThread);
	}

	private int getExactHeuristic(LpSolve solver, int marking, byte[] markingArray, int markingBlock, int markingIndex,
			double[] vars) {

		long start = System.nanoTime();
		// start from correct right hand side
		try {
			for (int p = net.numPlaces(); p-- > 0;) {
				// set right hand side to final marking 
				solver.setRh(p + 1, rhf[p] - markingArray[p]);
			}

			solver.defaultBasis();
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
				assert c >= 0;

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
				assert timeoutAtTimeInMillisecond - System.currentTimeMillis() <= 0;
				alignmentResult |= Utils.TIMEOUTREACHED;
				return HEURISTICINFINITE;
			} else {
				solver.writeLp("C:/temp/alignment/loopdouble_500K/debugLP-Alignment.lp");
				System.err.println("Error code from LpSolve solver:" + solverResult);
				System.exit(1);
				return HEURISTICINFINITE;
			}

		} catch (LpSolveException e) {
			return HEURISTICINFINITE;
		} finally {
			long st = System.nanoTime() - start;
			solveTime += st;
		}

	}

	protected double computeCostForVars(double[] vars) {
		double c = 0;
		for (int t = net.numTransitions(); t-- > 0;) {
			c += vars[t] * net.getCost(t);
		}
		return c;
	}

	protected void deriveOrEstimateHValue(int from, int fromBlock, int fromIndex, int transition, int to, int toBlock,
			int toIndex) {
		if (hasExactHeuristic(fromBlock, fromIndex) && getHScore(fromBlock, fromIndex) != HEURISTICINFINITE
				&& (getLpSolution(from, transition) >= 1)) {
			// from Marking has exact heuristic
			// we can derive an exact heuristic from it

			setDerivedLpSolution(from, to, transition);
			// set the exact h score
			setHScore(toBlock, toIndex, getHScore(fromBlock, fromIndex) - net.getCost(transition), true);
			heuristicsDerived++;

		} else if (hasExactHeuristic(fromBlock, fromIndex) && getHScore(fromBlock, fromIndex) == HEURISTICINFINITE) {
			// marking from which final state cannot be reached
			assert false;
			setHScore(toBlock, toIndex, HEURISTICINFINITE, true);
			heuristicsDerived++;
		} else {

			if (isFinal(to)) {
				setHScore(toBlock, toIndex, 0, true);
			}
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

	@Override
	public long getEstimatedMemorySize() {
		long val = super.getEstimatedMemorySize();
		// count size of rhs
		val += rhf.length * 8 + 4;
		return val;
	}

	@Override
	protected void fillStatistics(int[] alignment) {
		super.fillStatistics(alignment);
		putStatistic(Statistic.HEURISTICTIME, (int) (solveTime / 1000));
	}

	@Override
	protected void writeEndOfAlignmentDot(int[] alignment, int markingsReachedInRun, int closedActionsInRun) {
		for (int m = 0; m < markingsReached; m++) {
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
		debug.println(Debug.DOT, "}");
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
		lpSolutionsSize -= 12 + 4 + (removed != null ? removed.length : null); // object size
		lpSolutionsSize -= 1 + 4 + 8; // used flag + key + value pointer

	}

}