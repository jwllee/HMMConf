package nl.tue.alignment.algorithms.syncproduct.petrinet;

import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectIntHashMap;

public class ReducedPlace {

	private final int id;

	private final TObjectIntMap<ReducedTransition> input = new TObjectIntHashMap<>(3, 0.5f, Integer.MIN_VALUE);

	private final TObjectIntMap<ReducedTransition> output = new TObjectIntHashMap<>(3, 0.5f, Integer.MIN_VALUE);

	public ReducedPlace(int id) {
		this.id = id;

	}

	public String toString() {
		return String.format("p%3d", id);
	}

	public int hashCode() {
		return id;
	}

	public boolean equals(Object o) {
		return o != null && o instanceof ReducedPlace ? ((ReducedPlace) o).id == id : false;
	}

	public String toIdString() {
		return "p" + id;
	}

	public TObjectIntMap<ReducedTransition> getInput() {
		return input;
	}

	public void addToInput(ReducedTransition transition, int multiplicity) {
		input.put(transition, multiplicity);
	}

	public TObjectIntMap<ReducedTransition> getOutput() {
		return output;
	}

	public void addToOutput(ReducedTransition transition, int multiplicity) {
		output.put(transition, multiplicity);
	}

	public String toHTMLString() {
		return input.size() + "/" + output.size();
	}

}
