package nl.tue.alignment.algorithms.datastructures;

import gnu.trove.impl.HashFunctions;
import gnu.trove.impl.hash.THash;
import nl.tue.alignment.algorithms.ReplayAlgorithm;
import nl.tue.alignment.algorithms.VisitedSet;

public class VisitedHashSet implements VisitedSet {

	public static final int NEV = -1;

	private final ReplayAlgorithm algorithm;

	private final TIntHashSetForMarkings set;

	/**
	 * Initializes a closed set with the given initial capacity
	 * 
	 * @param algorithm
	 * @param initialCapacity
	 */
	public VisitedHashSet(ReplayAlgorithm algorithm, int initialCapacity) {
		this.algorithm = algorithm;
		this.set = new TIntHashSetForMarkings(algorithm, initialCapacity);
	}

	@Override
	public ReplayAlgorithm getAlgorithm() {
		return algorithm;
	}

	@Override
	public int add(byte[] marking, int newIndex) {
		return set.add(marking, newIndex);
	}

	public String toString() {
		return set.toString();
	}

	public int capacity() {
		return set.capacity();
	}

	public void clear() {
		set.clear();
	}

	/**
	 * returns the maximum memory use in bytes the queue ever had.
	 * 
	 * @return
	 */
	public long getEstimatedMemorySize() {
		return 16 + 33 + 4 + set.capacity() * 4;
	}
}

class TIntHashSetForMarkings extends THash {

	private final ReplayAlgorithm algorithm;

	private int[] set;

	protected boolean consumeFreeSlot;

	/** mask used to check if a slot is free */
	public static final int FREE = 0;

	/** mask used to check if a slot is full */
	public static final int FULL = 1 << 31;

	/** mask used to check if a slot is full */
	public static final int OLD = 1 << 31;

	public TIntHashSetForMarkings(ReplayAlgorithm algorithm, int initialCapacity) {
		super(initialCapacity, DEFAULT_LOAD_FACTOR);
		this.algorithm = algorithm;
		initialCapacity = Math.max(1, initialCapacity);
		_loadFactor = DEFAULT_LOAD_FACTOR;
		setUp(HashFunctions.fastCeil(initialCapacity / _loadFactor));

	}

	public int add(byte[] marking, int markingIndex) {
		int index = insertKey(marking, markingIndex);

		if ((index & OLD) == OLD) {
			return getValue(index & ~OLD); // already present in set, nothing to add
		}
		assert getValue(index) == markingIndex;

		algorithm.addNewMarking(marking);

		postInsertHook(consumeFreeSlot);

		return markingIndex; // yes, we added something
	}

	public boolean contains(byte[] marking) {
		int idx = index(marking);
		if (idx < 0) {
			return false;
		} else {
			return true;
		}
	}

	/**
	 * Locates the index at which <tt>val</tt> can be inserted. if there is already
	 * a value equal()ing <tt>val</tt> in the set, returns that value as a negative
	 * integer.
	 * 
	 * @param val
	 *            an <code>int</code> value
	 * @return an <code>int</code> value
	 */
	protected int insertKey(byte[] marking, int markingIndex) {
		int hash, index;

		hash = algorithm.hashCode(marking) & 0x7fffffff;
		index = hash % set.length;
		int state = set[index] & FULL;

		consumeFreeSlot = false;

		if (state == FREE) {
			consumeFreeSlot = true;
			this.insertKeyAt(index, markingIndex);

			return index; // empty, all done
		}

		if (state == FULL && algorithm.equalMarking(getValue(index), marking)) {
			return index | OLD; // already stored
		}

		// already FULL or REMOVED, must probe
		return this.insertKeyRehash(marking, index, hash, state, markingIndex);
	}

	private int insertKey(int markingIndex) {
		int hash, index;

		hash = algorithm.hashCode(markingIndex) & 0x7fffffff;
		index = hash % set.length;
		int state = set[index] & FULL;

		consumeFreeSlot = false;

		if (state == FREE) {
			consumeFreeSlot = true;
			this.insertKeyAt(index, markingIndex);

			return index; // empty, all done
		}

		if (state == FULL && getValue(index) == markingIndex) {
			return index | OLD; // already stored
		}

		// already FULL or REMOVED, must probe
		return this.insertKeyRehash(index, hash, state, markingIndex);
	}

	private int getValue(int index) {
		return set[index] & ~FULL;
	}

	private int getState(int index) {
		return set[index] & FULL;
	}

	int insertKeyRehash(byte[] marking, int index, int hash, int state, int markingIndex) {
		// compute the double hash
		final int length = set.length;
		int probe = 1 + (hash % (length - 2));
		final int loopIndex = index;

		/**
		 * Look until FREE slot or we start to loop
		 */
		do {

			index -= probe;
			if (index < 0) {
				index += length;
			}
			state = getState(index);

			// A FREE slot stops the search
			if (state == FREE) {

				consumeFreeSlot = true;
				this.insertKeyAt(index, markingIndex);
				return index;

			}

			if (state == FULL && algorithm.equalMarking(getValue(index), marking)) {
				return index | OLD; // already stored
			}

			// Detect loop
		} while (index != loopIndex);

		// Can a resizing strategy be found that resizes the set?
		throw new IllegalStateException("No free or removed slots available. Key set full?!!");
	}

	int insertKeyRehash(int index, int hash, int state, int markingIndex) {
		// compute the double hash
		final int length = set.length;
		int probe = 1 + (hash % (length - 2));
		final int loopIndex = index;

		/**
		 * Look until FREE slot or we start to loop
		 */
		do {

			index -= probe;
			if (index < 0) {
				index += length;
			}
			state = getState(index);

			// A FREE slot stops the search
			if (state == FREE) {

				consumeFreeSlot = true;
				this.insertKeyAt(index, markingIndex);
				return index;

			}

			if (state == FULL && getValue(index) == markingIndex) {
				return index | OLD; // already stored
			}

			// Detect loop
		} while (index != loopIndex);

		// Can a resizing strategy be found that resizes the set?
		throw new IllegalStateException("No free or removed slots available. Key set full?!!");
	}

	void insertKeyAt(int index, int markingIndex) {
		set[index] = (markingIndex & ~FULL) | FULL; // insert value and set mask to full
	}

	/**
	 * Locates the index of <tt>val</tt>.
	 * 
	 * @param val
	 *            an <code>int</code> value
	 * @return the index of <tt>val</tt> or -1 if it isn't in the set.
	 */
	protected int index(byte[] val) {
		int hash, index, length;

		length = set.length;
		hash = algorithm.hashCode(val) & 0x7fffffff;
		index = hash % length;
		int state = getState(index);

		if (state == FREE)
			return -1;

		if (state == FULL && algorithm.equalMarking(getValue(index), val))
			return index;

		return this.indexRehashed(val, index, hash, state);
	}

	int indexRehashed(byte[] key, int index, int hash, int state) {
		// see Knuth, p. 529
		int length = set.length;
		int probe = 1 + (hash % (length - 2));
		final int loopIndex = index;

		do {
			index -= probe;
			if (index < 0) {
				index += length;
			}
			state = getState(index);
			//
			if (state == FREE)
				return -1;

			//
			if (algorithm.equalMarking(getValue(index), key))
				return index;
		} while (index != loopIndex);

		return -1;
	}

	protected void rehash(int newCapacity) {
		int oldCapacity = set.length;

		int oldSet[] = set;

		set = new int[newCapacity];

		for (int i = oldCapacity; i-- > 0;) {
			if ((oldSet[i] & FULL) == FULL) {
				int o = oldSet[i] & ~FULL;
				insertKey(o);
			}
		}
	}

	@Override
	public void clear() {
		super.clear();
		for (int i = set.length; i-- > 0;) {
			set[i] = FREE;
		}

	}

	/**
	 * initializes the hashtable to a prime capacity which is at least
	 * <tt>initialCapacity + 1</tt>.
	 * 
	 * @param initialCapacity
	 *            an <code>int</code> value
	 * @return the actual capacity chosen
	 */
	protected int setUp(int initialCapacity) {
		int capacity = super.setUp(initialCapacity);
		set = new int[capacity];
		return capacity;
	}

	public int capacity() {
		return set.length;
	}

}
