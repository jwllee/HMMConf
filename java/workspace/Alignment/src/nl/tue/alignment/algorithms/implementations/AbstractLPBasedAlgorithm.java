package nl.tue.alignment.algorithms.implementations;

import java.util.Arrays;

import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import lpsolve.LpSolve;
import nl.tue.alignment.algorithms.syncproduct.SyncProduct;

abstract class AbstractLPBasedAlgorithm extends AbstractReplayAlgorithm {

	// for each stored solution, the first byte is used for flagging.
	// the first bit indicates whether the solution is derived
	// The next three bits store the number of bits per transition (0 implies 1 bit per transition, 7 implies 8 bits per transition)
	// The rest of the array  then stores the solution
	protected static final byte COMPUTED = (byte) 0b00000000;
	protected static final byte DERIVED = (byte) 0b10000000;

	protected static final byte BITPERTRANSMASK = (byte) 0b01111000;
	protected static final byte FREEBITSFIRSTBYTE = 3;
	protected int maxMapCapacity = 16;

	// stores the location of the LP solution plus a flag if it is derived or real
	protected TIntObjectMap<byte[]> lpSolutions = new TIntObjectHashMap<byte[]>(maxMapCapacity) {
		@Override
		protected void rehash(int newCapacity) {

			super.rehash(newCapacity);
			if (newCapacity > maxMapCapacity) {
				maxMapCapacity = newCapacity;
			}
		}

	};
	protected long lpSolutionsSize = 4;
	protected long bytesLpSolutionsSize = 4;

	protected LpSolve solver;
	protected int bytesUsed;
	protected long solveTime = 0;

	/**
	 * In the abstract LP Based algorithm, the translation is made from transitions
	 * to moveIDs using the net's getMove() methods.
	 * 
	 * @param net
	 * @param moveSorting
	 * @param queueSorting
	 * @param preferExact
	 * @param debug
	 */
	public AbstractLPBasedAlgorithm(SyncProduct net, boolean moveSorting, boolean queueSorting, boolean preferExact,
			Debug debug) {
		super(net, moveSorting, queueSorting, preferExact, debug);
		tempForSettingSolution = new int[net.numTransitions()];
	}

	protected int[] tempForSettingSolution;

	protected void setNewLpSolution(int marking, double[] solutionDouble) {
		// copy the solution from double array to byte array (rounding down)
		// and compute the maximum.
		Arrays.fill(tempForSettingSolution, 0);
		byte bits = 0;
		for (int i = tempForSettingSolution.length; i-- > 0;) {
			tempForSettingSolution[i] = ((int) (solutionDouble[i] + 1E-7));
			if (tempForSettingSolution[i] > (1 << bits)) {
				bits++;
			}
		}
		bits++;
		setNewLpSolution(marking, bits, tempForSettingSolution);
	}

	/**
	 * Stores the solution vector. This vector should be on the move level!
	 * 
	 * @param marking
	 * @param bits
	 * @param solutionInt
	 */
	protected void setNewLpSolution(int marking, int bits, int[] solutionInt) {

		// to store this solution, we need "bits" bits per transition
		// plus a header consisting of 8-FREEBITSFIRSTBYTE bits.
		// this translate to 
		int bytes = 8 - FREEBITSFIRSTBYTE + (solutionInt.length * bits + 4) / 8;

		//		assert getSolution(marking) == null;
		byte[] solution = new byte[bytes];

		// set the computed flag in the first two bits
		solution[0] = COMPUTED;
		// set the number of bits used in the following 3 bits
		bits--;
		solution[0] |= bits << FREEBITSFIRSTBYTE;

		int currentByte = 0;
		byte currentBit = (1 << (FREEBITSFIRSTBYTE - 1));
		for (int t = 0; t < solutionInt.length; t++) {
			// tempForSettingSolution[i] can be stored in "bits" bits.
			for (int b = 1 << bits; b > 0; b >>>= 1) {
				// copy the appropriate bit
				if ((solutionInt[t] & b) != 0)
					solution[currentByte] |= currentBit;

				// rotate right
				currentBit = (byte) ((((currentBit & 0xFF) >>> 1) | (currentBit << 7)));
				if (currentBit < 0)
					currentByte++;

			}
		}
		addSolution(marking, solution);
		//		for (int i = 0; i < net.numTransitions(); i++) {
		//			assert (solutionInt[translate(i)] == getLpSolution(marking, i)) : "Error in " + i;
		//		}
	}

	protected int getLpSolution(int marking, int transition) {

		byte[] solution = getSolution(marking);
		//		if ((solution[0] & STOREDFULL) == STOREDFULL) {
		// get the bits used per transition
		int bits = 1 + ((solution[0] & BITPERTRANSMASK) >>> FREEBITSFIRSTBYTE);
		// which is the first bit?
		int fromBit = 8 - FREEBITSFIRSTBYTE + transition * bits;
		// that implies the following byte
		int fromByte = fromBit >>> 3;
		// with the following index in byte.
		fromBit &= 7;

		byte currentBit = (byte) (1 << (7 - fromBit));
		int value = 0;
		for (int i = 0; i < bits; i++) {
			// shift value left
			value <<= 1;

			// flip the bit
			if ((solution[fromByte] & currentBit) != 0)
				value++;

			// rotate bit right 
			currentBit = (byte) (((currentBit & 0xFF) >>> 1) | (currentBit << 7));
			// increase byte if needed.
			if (currentBit < 0)
				fromByte++;

		}

		return value;
	}

	private byte[] getSolution(int marking) {
		return lpSolutions.get(marking);
	}

	protected boolean isDerivedLpSolution(int marking) {
		return getSolution(marking) != null && (getSolution(marking)[0] & DERIVED) == DERIVED;
	}

	protected void setDerivedLpSolution(int from, int to, int transition) {

		//		assert getSolution(to) == null;
		byte[] solutionFrom = getSolution(from);

		byte[] solution = Arrays.copyOf(solutionFrom, solutionFrom.length);

		solution[0] |= DERIVED;

		// get the length of the bits used per transition
		int bits = 1 + ((solution[0] & BITPERTRANSMASK) >>> FREEBITSFIRSTBYTE);
		// which is the least significant bit?
		int fromBit = 8 - FREEBITSFIRSTBYTE + transition * bits + (bits - 1);
		// that implies the following byte
		int fromByte = fromBit >>> 3;
		// with the following index in byte.
		fromBit &= 7;
		// most significant bit in fromBit
		byte lsBit = (byte) (1 << (7 - fromBit));

		// we need to reduce by 1.
		for (int i = 0; i < bits; i++) {
			// flip the bit
			if ((solution[fromByte] & lsBit) != 0) {
				// first bit that is 1. Flip and terminate
				solution[fromByte] ^= lsBit;
				addSolution(to, solution);
				//				assert getLpSolution(to, transition) == getLpSolution(from, transition) - 1;
				return;
			}
			// flip and continue;
			solution[fromByte] ^= lsBit;
			// rotate bit left
			lsBit = (byte) (((lsBit & 0xFF) >>> 7) | (lsBit << 1));
			// decrease byte if needed.
			if (lsBit == 1)
				fromByte--;

		}
		assert false;
		throw new RuntimeException("Unreachable Code Reached.");
	}

	private void addSolution(int marking, byte[] solution) {
		lpSolutions.put(marking, solution);
		lpSolutionsSize += 12 + 4 + solution.length; // object size
	}

	protected abstract double computeCostForVars(double[] vars);

	@Override
	public long getEstimatedMemorySize() {
		long val = super.getEstimatedMemorySize();
		// count space for all computed solutions
		val += bytesLpSolutionsSize;
		// count space for map
		val += maxMapCapacity * (1 + 4 + 8);
		// count size of matrix
		val += bytesUsed;
		return val;
	}

	/**
	 * In ILP version, only one given final marking is the target.
	 */
	@Override
	protected boolean isFinal(int marking) {
		return equalMarking(marking, net.getFinalMarking());
	}

	@Override
	protected void terminateIteration(int[] alignment, int markingsReachedInRun, int closedActionsInRun) {
		try {
			super.terminateIteration(alignment, markingsReachedInRun, closedActionsInRun);
		} finally {
			solver.deleteAndRemoveLp();
		}
	}

}
