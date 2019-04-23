package nl.tue.alignment.algorithms.syncproduct.petrinet;

import gnu.trove.map.hash.TObjectIntHashMap;

class Sequence2SyncMoveCost extends TObjectIntHashMap<TransitionEventClassList> {

	public Sequence2SyncMoveCost() {
		super(5, 0.5f, Integer.MAX_VALUE);
	}

}