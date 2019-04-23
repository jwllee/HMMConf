package nl.tue.alignment.algorithms.implementations;

import nl.tue.alignment.algorithms.ReplayAlgorithm;
import nl.tue.alignment.algorithms.ReplayAlgorithm.Debug;
import nl.tue.alignment.algorithms.syncproduct.SyncProduct;

/**
 * Implements a variant of Dijkstra's shortest path algorithm for alignments,
 * i.e. the heuristic is always equal to 0.
 * 
 * This implementation can only be used to compute the search space, as it will
 * not consider any marking final, i.e. it terminates when all markings have
 * been reached (or it runs out of memory)
 * 
 * @author bfvdonge
 * 
 */
public class FullStatespaceBuilder extends Dijkstra {

	public FullStatespaceBuilder(SyncProduct product, boolean moveSorting, boolean queueSorting,  
			Debug debug) {
		super(product, moveSorting, queueSorting,  debug);
	}

	@Override
	protected boolean isFinal(int marking) {
		return false;
	}

}