package nl.tue.alignment.algorithms;

import java.io.PrintStream;

import gnu.trove.map.TObjectIntMap;
import nl.tue.alignment.Canceler;
import nl.tue.alignment.Utils;
import nl.tue.alignment.Utils.Statistic;
import nl.tue.alignment.algorithms.syncproduct.SyncProduct;
import nl.tue.astar.util.ilp.LPMatrixException;

public interface ReplayAlgorithm {

	public enum CloseResult {
		CLOSEDSUCCESSFUL, CLOSEDINFEASIBLE, FINALMARKINGFOUND, REQUEUED, RESTARTNEEDED;
	}

	/**
	 * Debug levels for the replayer. Replay level NONE does not produce any output.
	 * Replay level NORMAL provides some output to the selected stream (normally
	 * System.out). Replay level STATS writes statistics in a standard format to the
	 * stream. This should be used with a filewriter stream to collect statistics in
	 * experiments. DOT writes a dot compatible output tot the stream. This can be
	 * used to obtain the search graphs and should be used for small graphs only.
	 * 
	 * @author bfvdonge
	 *
	 */
	public static enum Debug {

		DOT {
			@Override
			public void writeMarkingReached(ReplayAlgorithm algorithm, int marking, String extra) {
				int heur = algorithm.getHScore(marking);
				StringBuilder b = new StringBuilder();
				b.append("i" + algorithm.getIterationNumber());
				b.append("m");
				b.append(marking);
				b.append(" [label=<");
				if (algorithm.isClosed(marking)) {
					b.append("(m");
					b.append(marking);
					b.append(")");
				} else {
					b.append("m");
					b.append(marking);
				}
				b.append("<BR/>");
				b.append(Utils.asBag(algorithm.getMarking(marking), algorithm.getNet()));
				b.append("<BR/>g=");
				b.append(algorithm.getGScore(marking));
				b.append(",");
				b.append(algorithm.hasExactHeuristic(marking) ? "h" : "~h");
				b.append("=");
				b.append((algorithm.isInfinite(heur) ? "inf" : heur));
				b.append(">");
				if (!extra.isEmpty()) {
					b.append(",");
					b.append(extra);
				}
				b.append("];");
				synchronized (output) {
					output.println(b.toString());
				}
			}

			@Override
			public void writeEdgeTraversed(ReplayAlgorithm algorithm, int fromMarking, int transition, int toMarking,
					String extra) {
				StringBuilder b = new StringBuilder();
				b.append("i" + algorithm.getIterationNumber());
				b.append("m");
				b.append(fromMarking);
				b.append(" -> ");
				b.append("i" + algorithm.getIterationNumber());
				b.append("m");
				b.append(toMarking);
				b.append(" [");
				if (transition >= 0) {
					b.append("label=<<b>");
					//				b.append("t");
					//				b.append(transition);
					//				b.append("<br/>");
					b.append(algorithm.getNet().getTransitionLabel(transition));
					b.append("<br/>");
					b.append(algorithm.getNet().getCost(transition));
					b.append("</b>>");
					if (algorithm.getNet().getTypeOf(transition) == SyncProduct.SYNC_MOVE) {
						b.append(",fontcolor=forestgreen");
					} else if (algorithm.getNet().getTypeOf(transition) == SyncProduct.MODEL_MOVE) {
						b.append(",fontcolor=darkorchid1");
					} else if (algorithm.getNet().getTypeOf(transition) == SyncProduct.LOG_MOVE) {
						b.append(",fontcolor=goldenrod2");
					} else if (algorithm.getNet().getTypeOf(transition) == SyncProduct.TAU_MOVE) {
						b.append(",fontcolor=honeydew4");
					}
				}
				if (!extra.isEmpty()) {
					b.append(extra);
				}

				b.append("];");
				synchronized (output) {
					output.println(b.toString());
				}
			}
		}, //
		NORMAL, //
		NONE, STATS;

		private static String EMPTY = "";
		private static PrintStream output = System.out;

		/**
		 * Should be called when an edge is traversed in the search from marking with ID
		 * fromMarking to marking with ID toMarking though firing transition.
		 * 
		 * @param algorithm
		 * @param fromMarking
		 * @param transition
		 * @param toMarking
		 * @param extra
		 */
		public void writeEdgeTraversed(ReplayAlgorithm algorithm, int fromMarking, int transition, int toMarking,
				String extra) {
		}

		public void writeMarkingReached(ReplayAlgorithm algorithm, int marking, String extra) {
		}

		public void writeEdgeTraversed(ReplayAlgorithm algorithm, int fromMarking, int transition, int toMarking) {
			this.writeEdgeTraversed(algorithm, fromMarking, transition, toMarking, EMPTY);
		}

		public void writeMarkingReached(ReplayAlgorithm algorithm, int marking) {
			this.writeMarkingReached(algorithm, marking, EMPTY);
		}

		public void println(Debug db, String s) {
			if (this == db) {
				synchronized (output) {
					output.println(s);
				}
			}
		}

		public void println(Debug db) {
			if (this == db) {
				synchronized (output) {
					output.println();
				}
			}
		}

		public void print(Debug db, String s) {
			if (this == db) {
				synchronized (output) {
					output.print(s);
				}
			}
		}

		/**
		 * Sets the output stream. User can change this from System.out to capture file
		 * output.
		 * 
		 * @param out
		 */
		public synchronized static void setOutputStream(PrintStream out) {
			output = out;
		}

		/**
		 * Returns the output stream currently set.
		 * 
		 * @return
		 */
		public synchronized static PrintStream getOutputStream() {
			return output;
		}
	}

	/**
	 * Returns the synchronous product for which this ReplayAlgorithm was
	 * instantiated
	 * 
	 * @return the synchronous product
	 */
	public SyncProduct getNet();

	/**
	 * Returns the current iteration number of the algorithm if there are multiple
	 * iterations in which the same marking IDs will be present. This method is
	 * needed for DOT output to separate the nodes in the iterations.
	 * 
	 * @return the iteration number
	 */
	public int getIterationNumber();

	/**
	 * Obtain the statistics after computing an alignment. Should only be called
	 * after a call to run();
	 * 
	 * @return
	 */
	public TObjectIntMap<Utils.Statistic> getStatistics();

	/**
	 * Run the replay algorithm. The canceler is provided for cancellation purposes.
	 * A timeout is specified in milliseconds and a maximum number of states is also
	 * provided. Furthermore, as cost upper limit can be given to bound the search.
	 * 
	 * @param canceler
	 *            Allows for canceling the computation
	 * @param timeoutMilliseconds
	 *            Timeout for the computation
	 * @param maximumNumberOfStates
	 *            Maximum number of states expanded
	 * @param costUpperLimit
	 *            Maximum cost of the alignment. Providing an upperbound of 0 will
	 *            answer the yes/no question if the trace is fitting (assuming
	 *            synchronous and tau-moves have cost 0)
	 * @return alignment as a sequence of transition firings from the initial
	 *         marking to (1) the final marking if the computation is succesful or
	 *         (2) any marking if it is not succesful
	 * @throws LPMatrixException
	 *             Exceptions is thrown if there is a problem with the LP
	 *             computations.
	 */
	public int[] run(Canceler canceler, int timeoutMilliseconds, int maximumNumberOfStates, int costUpperLimit)
			throws LPMatrixException;

	/**
	 * Adds a statistic to the set of statistics. Can be called before or after
	 * run(), but the run() method may overwrite previously added statistics.
	 * 
	 * @param stat
	 * @param value
	 */
	public void putStatistic(Statistic stat, int value);

	/**
	 * Adds a new marking to the internal storage. Method should only be called if
	 * it does not exist already.
	 * 
	 * @param marking
	 *            the byte[] indicating the number of tokens per place
	 * @return the id of the marking in the internal storage.
	 */
	public int addNewMarking(byte[] marking);

	/**
	 * Returns the explicit representation of the marking stored with ID markingId.
	 * 
	 * @param markingId
	 *            an ID of a previously stored marking (ID obtained through
	 *            addNewMarking())
	 * @return the explicit representation of the marking
	 */
	public byte[] getMarking(int markingId);

	/**
	 * Returns the hashCode for a marking represented explicitly as an array of
	 * bytes.
	 * 
	 * @param marking
	 *            the byte[] indicating the number of tokens per place
	 * @return a (non-cryptographic) hashcode for this marking
	 */
	public int hashCode(byte[] marking);

	/**
	 * Check for equality between the stored marking with ID markingId and an
	 * explicit marking.
	 * 
	 * @param markingId
	 *            an ID of a previously stored marking (ID obtained through
	 *            addNewMarking())
	 * @param marking
	 *            the byte[] indicating the number of tokens per place
	 * @return a boolean indicating equality.
	 */
	public boolean equalMarking(int markingId, byte[] marking);

	/**
	 * Returns the hashCode for the stored marking with ID markingId.
	 * 
	 * @param markingId
	 *            an ID of a previously stored marking (ID obtained through
	 *            addNewMarking()
	 * @return a (non-cryptographic) hashcode for this marking
	 */
	public int hashCode(int markingId);

	/**
	 * Checks if the stored marking with ID markingId has an exact value for the
	 * heuristic function g.
	 * 
	 * @param markingId
	 *            an ID of a previously stored marking (ID obtained through
	 *            addNewMarking()
	 * @return true if the stored value for g is exact.
	 */
	public boolean hasExactHeuristic(int markingId);

	/**
	 * Returns the highest rank of an event explained by the path through which the
	 * stored marking with ID markingId was reached.
	 * 
	 * @param markingId
	 *            an ID of a previously stored marking (ID obtained through
	 *            addNewMarking()
	 * @return the highest rank of any event explained by a path to this marking.
	 */
	public int getLastRankOf(int markingId);

	/**
	 * Returns the f score (i.e. the sum of the g and h score) for the stored
	 * marking with ID markingId .
	 * 
	 * @param markingId
	 *            an ID of a previously stored marking (ID obtained through
	 *            addNewMarking()
	 * @return The sum of g and h score.
	 */
	public int getFScore(int markingId);

	/**
	 * Returns the g score (i.e. the minimal cost for reaching this marking so-far)
	 * for the stored marking with ID markingId .
	 * 
	 * @param markingId
	 *            an ID of a previously stored marking (ID obtained through
	 *            addNewMarking()
	 * @return The minimum cost through which this marking can be reached at this
	 *         point.
	 */
	public int getGScore(int markingId);

	/**
	 * Returns the h score (i.e. an underestimate for the remaining cost to reach
	 * the final marking) for the stored marking with ID markingId .
	 * 
	 * @param markingId
	 *            an ID of a previously stored marking (ID obtained through
	 *            addNewMarking()
	 * @return The current underestimate for the remaining cost of reaching the
	 *         final marking.
	 */
	public int getHScore(int marking);

	/**
	 * Checks if a value for the h-score is INFINITE.
	 * 
	 * @param heur
	 *            a value for the h-score
	 * @return true if the provided value is considered INFINITE
	 */
	public boolean isInfinite(int heur);

	/**
	 * Checks if the stored marking with ID markingId is in the closed Set.
	 * 
	 * @param markingId
	 *            an ID of a previously stored marking (ID obtained through
	 *            addNewMarking()
	 * @return true if the marking is in the closed set.
	 */
	public boolean isClosed(int markingId);

	/**
	 * Estimates the memory size of the internal data structures in bytes.
	 * 
	 * @return The estimated memory size of the internal data structures in bytes.
	 */
	public long getEstimatedMemorySize();

	/**
	 * returns the length of the path from the initial marking to reach this marking
	 * 
	 * @param marking
	 * @return
	 */
	public int getPathLength(int marking);

}