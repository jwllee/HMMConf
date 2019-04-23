package nl.tue.alignment.algorithms.syncproduct.petrinet;

import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import gnu.trove.procedure.TObjectIntProcedure;

public class ReducedTransition {

	private static int nextID = 0;

	private final int id;

	private final int maxSequenceLength;

	public static final int INHIBITOR = -1;

	//public static final int RESET = -2;

	public enum Type {
		BASIC, SEQUENCE, CHOICE;
	}

	/**
	 * Merges transitions based on the merge type given.
	 * 
	 * @param type
	 *            should not be null!
	 * @param sources
	 *            source transitions to merge.
	 * @return
	 */
	public static ReducedTransition merge(Type type, ReducedTransition... sources) {
		if (type == null) {
			throw new UnsupportedOperationException("cannot merge unknown type.");
		}
		if (sources.length == 1) {
			return sources[0];
		}
		switch (type) {
			case CHOICE :
				return mergeChoice(sources);
			case SEQUENCE :
				return mergeSequence(sources);
			default :
				throw new UnsupportedOperationException("cannot merge this type: " + type);
		}
	}

	/**
	 * Constructs a new reduced transition which is the exclusive choice merge of
	 * the input
	 * 
	 * @param sources
	 * @return
	 */
	public static ReducedTransition mergeChoice(ReducedTransition... sources) {
		int len = 0;
		for (ReducedTransition s : sources) {
			if (s.maxSequenceLength > len) {
				len = s.maxSequenceLength;
			}
		}

		final ReducedTransition result = new ReducedTransition(Type.CHOICE, len);

		for (ReducedTransition s : sources) {
			// in case of a choice, copy all source sequences with respective cost.
			s.sequence2cost.forEachEntry(new TObjectIntProcedure<TransitionEventClassList>() {

				public boolean execute(TransitionEventClassList a, int b) {
					// copy the sequence, but only if the new cost are smaller
					if (result.sequence2cost.get(a) > b) {
						// remove key to ensure replacement of the key by "a"
						result.sequence2cost.remove(a);
						// put key "a" back in
						result.sequence2cost.put(a, b);
					}
					return true;
				}
			});

		}
		// add the input and output places of each transition.
		result.input.putAll(sources[0].input);
		result.output.putAll(sources[0].output);

		return result;
	}

	/**
	 * Constructs a new reduced transition which is the sequential merge of the
	 * input
	 * 
	 * @param sources
	 * @return
	 */
	public static ReducedTransition mergeSequence(ReducedTransition... sources) {
		int len = 0;
		for (ReducedTransition s : sources) {
			len += s.maxSequenceLength;
		}

		final ReducedTransition result = new ReducedTransition(Type.SEQUENCE, len);

		for (ReducedTransition s : sources) {
			// move on model cost is the sum over the sources
			result.addSequences(s);
		}

		// add the input and output places of the sequence
		result.input.putAll(sources[0].input);
		result.output.putAll(sources[sources.length - 1].output);

		return result;
	}

	/**
	 * Checks if the sources can be merged and if so returns the merge type.
	 * 
	 * @param sources
	 * @return Type.BASIC if merging cannot be done (for whatever reason)
	 */
	public static Type canMerge(int maxLength, ReducedTransition... sources) {
		if (sources.length <= 1 || maxLength <= 1) {
			return Type.BASIC;
		}
		TObjectIntMap<ReducedPlace> input = sources[0].input;
		TObjectIntMap<ReducedPlace> output = sources[0].output;

		Type type = null;
		int len = sources[0].maxSequenceLength;
		for (int i = 1; i < sources.length; i++) {
			if (sources[i].input.equals(output) && (type == Type.SEQUENCE || type == null)) {
				// input matches the output of previous transition
				// ensure there is no other transition consuming from any of the places
				// in between.
				len += sources[i].maxSequenceLength;
				if (len > maxLength) {
					type = null;
					// potential sequence pattern, but maxlength too large.
					break;
				}
				type = Type.SEQUENCE;
				for (Object o : sources[i].input.keys()) {
					ReducedPlace rp = (ReducedPlace) o;
					if (rp.getInput().size() > 1 || rp.getOutput().size() > 1) {
						type = null;
						// potential sequence pattern, but intermediate input/output
						break;
					}
				}
			} else if (sources[i].input.equals(input) && sources[i].output.equals(output)
					&& (type == Type.CHOICE || type == null)) {
				if (sources[i].maxSequenceLength <= maxLength) {
					type = Type.CHOICE;
				} else {
					// potential choice pattern, but maxlength too large
					type = null;
					break;
				}
			} else {
				// Cannot match
				return Type.BASIC;
			}
			output = sources[i].output;
		}

		// match of correct type found.
		return type == null ? Type.BASIC : type;
	}

	/**
	 * Split a list of transitionEventClasses into a list of reduced transitions,
	 * each corresponding to exactly one event. Costs are attributed to the first
	 * transition.
	 * 
	 * @param list
	 * @param syncMoveCost
	 * @return
	 */
	public static ReducedTransition[] createList(TransitionEventClassList list, int syncMoveCost, int modelMoveCost) {
		ReducedTransition[] result = new ReducedTransition[list.getEventClassSequence().length];
		int[] transitions = list.getTransitionSequence();
		int[] events = list.getEventClassSequence();
		int start = 0;
		int it = 0, ik = 0;
		do {
			while (it < transitions.length && (transitions[it] < 0 || ik == events.length - 1)) {
				// skip the initial model moves
				it++;
			}
			if (it < transitions.length) {
				// not yet the last one.
				it++;
			}
			result[ik] = new ReducedTransition(Type.SEQUENCE, it - start);
			result[ik].sequence2cost.put(list.subList(events[ik], start, it), ik == 0 ? syncMoveCost : 0);
			result[ik].sequence2cost.put(TransitionEventClassList.EMPTY, ik == 0 ? modelMoveCost : 0);

			start = it;
			ik++;
		} while (ik < events.length);

		return result;
	}

	private final Type type;

	// each reduced transition corresponds to an exclusive choice of a variety of sequences of original 
	// transitions. Each with specific cost for syncing exactly that sequence of transitions 
	public Sequence2SyncMoveCost sequence2cost = new Sequence2SyncMoveCost();

	private final TObjectIntMap<ReducedPlace> input = new TObjectIntHashMap<>(3, 0.5f, Integer.MIN_VALUE);

	private final TObjectIntMap<ReducedPlace> output = new TObjectIntHashMap<>(3, 0.5f, Integer.MIN_VALUE);

	public ReducedTransition(int transitionId, int eventClassId, int modelMoveCost, int syncMoveCost) {
		synchronized (ReducedTransition.class) {
			this.id = nextID++;
		}
		maxSequenceLength = 1;
		// transition can be mapped to sequence <id> with cost minimum of existing and syncmove cost on s.
		sequence2cost.put(new TransitionEventClassList(transitionId, eventClassId), syncMoveCost);
		sequence2cost.put(new TransitionEventClassList(transitionId), modelMoveCost);
		this.type = Type.BASIC;
	}

	public ReducedTransition(int transitionId, int modelMoveCost) {
		synchronized (ReducedTransition.class) {
			this.id = nextID++;
		}
		maxSequenceLength = 1;
		// tau transition..
		sequence2cost.put(new TransitionEventClassList(transitionId), modelMoveCost);
		this.type = Type.BASIC;
	}

	private ReducedTransition(Type type, int maxSequenceLength) {
		synchronized (ReducedTransition.class) {
			this.id = nextID++;
		}
		this.type = type;
		this.maxSequenceLength = maxSequenceLength;
		//		sequence2cost.put(TransitionEventClassList.EMPTY, 0);
	}

	private void addSequences(ReducedTransition t) {
		final Sequence2SyncMoveCost newSequences = new Sequence2SyncMoveCost();

		if (sequence2cost.isEmpty()) {
			newSequences.putAll(t.sequence2cost);
		} else {

			// for each entry in the transition to add.
			t.sequence2cost.forEachEntry(new TObjectIntProcedure<TransitionEventClassList>() {

				// for each entry in the existing sequence list
				public boolean execute(final TransitionEventClassList a, final int b) {

					sequence2cost.forEachEntry(new TObjectIntProcedure<TransitionEventClassList>() {

						public boolean execute(TransitionEventClassList c, int d) {

							// newSeq is the concatenation of c and a.
							TransitionEventClassList newSeq = new TransitionEventClassList(c, a);

							// store the minimum cost of doing a sync move sequence on them
							newSequences.put(newSeq, b + d);
							// continue with the next.
							return true;
						}
					});
					// continue with the next.
					return true;
				}
			});
		}
		this.sequence2cost = newSequences;
	}

	public Type getType() {
		return type;
	}

	public TObjectIntMap<ReducedPlace> getInput() {
		return input;
	}

	public void addToInput(ReducedPlace place, int multiplicity) {
		input.put(place, multiplicity);
	}

	public TObjectIntMap<ReducedPlace> getOutput() {
		return output;
	}

	public void addToOutput(ReducedPlace place, int multiplicity) {
		output.put(place, multiplicity);
	}

	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("rt");
		builder.append(id);
		builder.append("\n");
		for (TransitionEventClassList seq : sequence2cost.keySet()) {
			builder.append(seq.toString());
			builder.append(":");
			builder.append(sequence2cost.get(seq));
			builder.append("\n");
		}
		return builder.toString();
	}

	public String toHTMLString() {
		StringBuilder builder = new StringBuilder();
		builder.append("rt");
		builder.append(id);
		builder.append("<BR/>");
		for (TransitionEventClassList seq : sequence2cost.keySet()) {
			builder.append(seq.toString());
			builder.append(" : ");
			builder.append(sequence2cost.get(seq));
			builder.append("<BR/>");
		}
		return builder.toString();
	}

	public String toIdString() {
		return "rt" + id;
	}

	public void forEachOutputArc(TObjectIntProcedure<ReducedPlace> procedure) {
		output.forEachEntry(procedure);
	}

	public void forEachInputArc(TObjectIntProcedure<ReducedPlace> procedure) {
		input.forEachEntry(procedure);
	}

	public int hashCode() {
		return id;
	}

	public boolean equals(Object o) {
		return o != null && o instanceof ReducedTransition ? ((ReducedTransition) o).id == id : false;
	}

	public int getModelMoveCost() {
		return sequence2cost.get(TransitionEventClassList.EMPTY);
	}

	public boolean mapsTo(int[] seq) {
		return sequence2cost.containsKey(new TransitionEventClassList.Wrap(seq));
	}

	public int getCostFor(int[] seq) {
		return sequence2cost.get(new TransitionEventClassList.Wrap(seq));
	}

	public void forEachSynchronousSequence(TObjectIntProcedure<TransitionEventClassList> procedure) {
		sequence2cost.forEachEntry(procedure);
	}

}