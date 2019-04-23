package nl.tue.alignment.algorithms.implementations;

import nl.tue.alignment.Utils;
import nl.tue.alignment.algorithms.syncproduct.SyncProduct;

/**
 * Implements a variant of Dijkstra's intest path algorithm for alignments, i.e.
 * the heuristic is always equal to 0.
 * 
 * This implementation can be used for prefix alignments, as it defers the
 * decision if a marking is a final marking to the synchronous product used as
 * input.
 * 
 * @author bfvdonge
 * 
 */
public class Dijkstra extends AbstractReplayAlgorithm {

	/**
	 * Special version of Dijkstra that builds the entire search space, i.e. this
	 * version never terminates correctly, but the search simply stops when the
	 * queueis empty.
	 * 
	 * @author bfvdonge
	 *
	 */
	public static class Full extends Dijkstra {

		private int firstFinal = -1;

		public Full(SyncProduct product) {
			super(product);
		}

		public Full(SyncProduct product, boolean moveSorting, boolean queueSorting, Debug debug) {
			super(product, moveSorting, queueSorting, debug);
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

	public Dijkstra(SyncProduct product) {
		this(product, false, false, Debug.NONE);
	}

	public Dijkstra(SyncProduct product, boolean moveSorting, boolean queueSorting, Debug debug) {
		super(product, moveSorting, queueSorting, true, debug);
		tempFinalMarking = new byte[numPlaces];
	}

	/**
	 * Dijkstra always estimates 0
	 */
	@Override
	public int getExactHeuristic(int marking, byte[] markingArray, int markingBlock, int markingIndex) {
		return 0;
	}

	private final transient byte[] tempFinalMarking;

	/**
	 * To allow for prefix versions of this algorithm, ask the net if the given
	 * marking is final.
	 */
	@Override
	protected boolean isFinal(int marking) {
		fillMarking(tempFinalMarking, marking);
		return net.isFinalMarking(tempFinalMarking);
	}

	protected void deriveOrEstimateHValue(int from, int fromBlock, int fromIndex, int transition, int to, int toBlock,
			int toIndex) {
		setHScore(toBlock, toIndex, 0, true);
	}

}