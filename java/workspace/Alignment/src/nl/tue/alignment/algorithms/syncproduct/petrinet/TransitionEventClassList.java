package nl.tue.alignment.algorithms.syncproduct.petrinet;

import java.util.Arrays;

/**
 * Keeps two lists. First a list of transition firings belonging to a reduced
 * transition, second a list of event classes (potentially shorter)
 * 
 * @author bfvdonge
 *
 */
public class TransitionEventClassList {
	private static int[] EMPTYARRAY = new int[0];

	public static class Wrap extends TransitionEventClassList {
		public Wrap(int[] eventClasses) {
			super(EMPTYARRAY, eventClasses);
		}
	}

	public static final TransitionEventClassList EMPTY = new TransitionEventClassList();

	private final int[] transitions;
	private final int[] eventClasses;

	private TransitionEventClassList(int[] transitions, int[] eventClasses) {
		this.transitions = transitions;
		this.eventClasses = eventClasses;
	}

	private TransitionEventClassList() {
		this(EMPTYARRAY, EMPTYARRAY);
	}

	public TransitionEventClassList(int transition) {
		// model move, store as negative number
		transitions = new int[] { -transition - 1 };
		eventClasses = new int[] {};
	}

	public TransitionEventClassList(int transition, int eventClass) {
		// sync move, store as non-negative number.
		transitions = new int[] { transition };
		eventClasses = new int[] { eventClass };
	}

	public TransitionEventClassList(TransitionEventClassList first, TransitionEventClassList second) {
		transitions = new int[first.transitions.length + second.transitions.length];
		eventClasses = new int[first.eventClasses.length + second.eventClasses.length];
		System.arraycopy(first.transitions, 0, transitions, 0, first.transitions.length);
		System.arraycopy(second.transitions, 0, transitions, first.transitions.length, second.transitions.length);
		System.arraycopy(first.eventClasses, 0, eventClasses, 0, first.eventClasses.length);
		System.arraycopy(second.eventClasses, 0, eventClasses, first.eventClasses.length, second.eventClasses.length);
	}

	public int hashCode() {
		return Arrays.hashCode(eventClasses);
	}

	public boolean equals(Object o) {
		return (o instanceof TransitionEventClassList
				? Arrays.equals(eventClasses, ((TransitionEventClassList) o).eventClasses)
				: false);
	}

	public String toString() {
		StringBuilder b = new StringBuilder();
		b.append('[');
		for (int i = 0; i < eventClasses.length; i++) {
			b.append(eventClasses[i]);
			b.append(",");
		}
		return b.append(']').toString();
	}

	public boolean endsWith(int cid) {
		return eventClasses.length > 0 && eventClasses[eventClasses.length - 1] == cid;
	}

	public int[] getEventClassSequence() {
		return eventClasses;
	}

	public int[] getTransitionSequence() {
		int[] result = new int[transitions.length];
		for (int i = result.length; i-- > 0;) {
			if (transitions[i] > 0) {
				result[i] = transitions[i];
			} else {
				result[i] = -transitions[i] - 1;
			}
		}
		return result;
	}

	public TransitionEventClassList subList(int eventClass, int from, int to) {
		return new TransitionEventClassList(Arrays.copyOfRange(transitions, from, to), new int[] { eventClass });
	}

	public int getTransitionSequenceLength() {
		return transitions.length;
	}

}