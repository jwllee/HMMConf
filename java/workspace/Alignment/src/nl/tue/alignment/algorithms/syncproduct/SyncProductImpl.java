package nl.tue.alignment.algorithms.syncproduct;

import java.util.Arrays;

public class SyncProductImpl implements SyncProduct {

	private static final int[] EMPTY = new int[0];

	protected final String[] transitions;

	protected final int[][] eventNumbers;

	protected final int[] ranks;

	protected final String[] places;

	protected final int[] cost;

	protected final int[][] input;

	protected final int[][] output;

	protected byte[] initMarking;

	protected byte[] finalMarking;

	protected final String label;

	private final byte[] types;

	private final int numEvents;

	private final int numClasses;

	private final int numModelMoves;

	private final int[] pathLengths;

	//	public SyncProductImpl(String label, int numClasses, String[] transitions, String[] places, int[][] eventNumbers,
	//			byte[] types, int[] moves, int[] cost) {
	//		this(label, numClasses, transitions, places, eventNumbers, eventNumbers, types, moves, cost);
	//	}

	public SyncProductImpl(String label, int numClasses, String[] transitions, String[] places, int[][] eventNumbers,
			int[] ranks, int[] pathLengths, byte[] types, int[] cost) {
		this.eventNumbers = eventNumbers;
		this.label = label;
		this.transitions = transitions;
		this.places = places;
		this.pathLengths = pathLengths;
		this.types = types;
		this.cost = cost;
		this.ranks = ranks;

		int mx = 0;
		for (int e = 0; e < eventNumbers.length; e++) {
			if (eventNumbers[e] != NOEVENT && eventNumbers[e][eventNumbers[e].length - 1] > mx) {
				mx = eventNumbers[e][eventNumbers[e].length - 1];
			}
		}
		this.numEvents = (mx + 1);

		mx = 0;
		for (int e = 0; e < types.length; e++) {
			if (types[e] == MODEL_MOVE || types[e] == TAU_MOVE) {
				mx++;
			}
		}
		this.numClasses = numClasses;
		this.numModelMoves = mx;

		input = new int[numTransitions()][];
		output = new int[numTransitions()][];

		initMarking = new byte[numPlaces()];
		finalMarking = new byte[numPlaces()];

	}

	public int numTransitions() {
		return transitions.length;
	}

	public int numPlaces() {
		return places.length;
	}

	private void setSortedArray(int[] array, int[] plist) {
		Arrays.sort(plist);
		for (int i = plist.length; i-- > 0;) {
			array[i] = plist[i];
		}
	}

	public void setInput(int t, int... plist) {
		input[t] = new int[plist.length];
		setSortedArray(input[t], plist);
	}

	public void setOutput(int t, int... plist) {
		output[t] = new int[plist.length];
		setSortedArray(output[t], plist);
	}

	public void addToOutput(int t, int... p) {
		output[t] = Arrays.copyOf(getOutput(t), getOutput(t).length + p.length);
		System.arraycopy(p, 0, output[t], output[t].length - p.length, p.length);
		Arrays.sort(output[t]);
	}

	public void addToInput(int t, int... p) {
		input[t] = Arrays.copyOf(getInput(t), getInput(t).length + p.length);
		System.arraycopy(p, 0, input[t], input[t].length - p.length, p.length);
		Arrays.sort(input[t]);
	}

	public int[] getInput(int transition) {
		return input[transition] == null ? EMPTY : input[transition];
	}

	public int[] getOutput(int transition) {
		return output[transition] == null ? EMPTY : output[transition];
	}

	public byte[] getInitialMarking() {
		return initMarking;
	}

	public byte[] getFinalMarking() {
		return finalMarking;
	}

	public void setInitialMarking(byte[] marking) {
		this.initMarking = marking;
	}

	public void setFinalMarking(byte[] marking) {
		this.finalMarking = marking;
	}

	public void setInitialMarking(int... places) {
		this.initMarking = new byte[numPlaces()];
		addToInitialMarking(places);
	}

	public void addToInitialMarking(int... places) {
		for (int p : places) {
			initMarking[p]++;
		}
	}

	public void setFinalMarking(int... places) {
		this.finalMarking = new byte[numPlaces()];
		addToFinalMarking(places);
	}

	public void addToFinalMarking(int... places) {
		for (int p : places) {
			finalMarking[p]++;
		}
	}

	public int getCost(int t) {
		return cost[t];
	}

	public String getTransitionLabel(int t) {
		return transitions[t];
	}

	public String getPlaceLabel(int p) {
		return places[p];
	}

	public void setTransitionLabel(int t, String label) {
		transitions[t] = label;
	}

	public void setPlaceLabel(int p, String label) {
		places[p] = label;
	}

	/**
	 * for full alignments: return Arrays.equals(marking, finalMarking);
	 * 
	 * for prefix alignments: check only if a specific place is marked.
	 * 
	 * For examples: place 18 marked with a single token return (marking[18 / 8] &
	 * (Utils.FLAG >>> (18 % 8))) != 0 && (marking[bm + 18 / 8] & (Utils.FLAG >>>
	 * (18 % 8))) == 0;
	 */
	public boolean isFinalMarking(byte[] marking) {
		// for full alignments:
		return Arrays.equals(marking, finalMarking);

	}

	public String getLabel() {
		return label;
	}

	public int[] getEventOf(int transition) {
		return eventNumbers[transition];
	}

	public void setEventOf(int transition, int[] event) {
		eventNumbers[transition] = event;
	}

	public byte getTypeOf(int transition) {
		return types[transition];
	}

	public int numEvents() {
		return numEvents;
	}

	public int getRankOf(int transition) {
		return ranks[transition];
	}

	public void setRankOf(int transition, int rank) {
		ranks[transition] = rank;
	}

	public int numEventClasses() {
		return numClasses;
	}

	public int numModelMoves() {
		return numModelMoves;
	}

	public int getTransitionPathLength(int transition) {
		return pathLengths[transition];
	}

}
