package nl.tue.alignment.algorithms.datastructures;

import java.util.Arrays;
import java.util.Collection;

import gnu.trove.map.hash.TIntIntHashMap;
import nl.tue.alignment.algorithms.Queue;
import nl.tue.alignment.algorithms.ReplayAlgorithm;

public class HashBackedPriorityQueue implements Queue {

	protected final ReplayAlgorithm algorithm;

	protected final TIntIntHashMap locationMap;
	protected static final int NEV = -1;

	/**
	 * Priority queue represented as a balanced binary heap: the two children of
	 * queue[n] are queue[2*n+1] and queue[2*(n+1)]. The priority queue is ordered
	 * by the record's natural ordering: For each node n in the heap and each
	 * descendant d of n, n <= d. The element with the best value is in queue[0],
	 * assuming the queue is nonempty.
	 */
	protected int[] queue;

	/**
	 * Stores the maximum capacity this queue ever had.
	 */
	private int maxQueueLength;

	/**
	 * Stores the maximum size this queue ever had.
	 */
	private int maxQueueSize;

	/**
	 * The number of elements in the priority queue.
	 */
	protected int size = 0;

	/**
	 * The maximum total cost for any record in this queue. If the cost of a record
	 * which is added is higher that this value, it is not added
	 */
	protected int maxCost;

	private final int initialCapacity;
	private int maxMapCapacity;

	public HashBackedPriorityQueue(ReplayAlgorithm algorithm, int initialCapacity) {
		this(algorithm, initialCapacity, Integer.MAX_VALUE);
	}

	public HashBackedPriorityQueue(ReplayAlgorithm algorithm, int initialCapacity, int maxCost) {
		this.algorithm = algorithm;
		this.initialCapacity = initialCapacity;
		this.maxCost = maxCost;
		maxMapCapacity = initialCapacity;
		locationMap = new TIntIntHashMap(initialCapacity, 0.5f, NEV, NEV) {
			@Override
			protected void rehash(int newCapacity) {

				super.rehash(newCapacity);
				if (newCapacity > maxMapCapacity) {
					maxMapCapacity = newCapacity;
				}
			}

		};
		this.queue = new int[initialCapacity];
		maxQueueLength = initialCapacity;
	}

	@Override
	public ReplayAlgorithm getAlgorithm() {
		return algorithm;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see nl.tue.astar.util.FastLookupPriorityQueue#isEmpty()
	 */
	public boolean isEmpty() {
		return size == 0;
	}

	public int hashCode() {
		return locationMap.hashCode();
	}

	public void setMaxCost(int maxCost) {
		this.maxCost = maxCost;
	}

	public int getMaxCost() {
		return this.maxCost;
	}

	public boolean contains(int marking) {
		return (locationMap.get(marking) != NEV);
	}

	public boolean checkInv() {
		return checkInv(0);
	}

	/**
	 * Increases the capacity of the array.
	 * 
	 * @param minCapacity
	 *            the desired minimum capacity
	 */
	protected void grow(int minCapacity) {
		if (minCapacity < 0) // overflow
			throw new OutOfMemoryError();
		int oldCapacity = queue.length;
		// Double size if small; else grow by 50%
		int newCapacity = ((oldCapacity < 64) ? ((oldCapacity + 1) * 2) : ((oldCapacity / 2) * 3));
		if (newCapacity < 0) // overflow
			newCapacity = Integer.MAX_VALUE;
		if (newCapacity < minCapacity)
			newCapacity = minCapacity;
		queue = Arrays.copyOf(queue, newCapacity);
		if (newCapacity > maxQueueLength) {
			maxQueueLength = newCapacity;
		}
	}

	public int peek() {
		if (size == 0)
			return NEV;
		return queue[0];
	}

	public int size() {
		return size;
	}

	public int poll() {
		if (size == 0)
			return NEV;
		int s = --size;
		int result = queue[0];
		int x = queue[s];
		queue[s] = -1;
		locationMap.remove(result);

		if (s != 0)
			siftDown(0, x);

		// shrink queue
		if (queue.length > size * 2) {
			if (size < initialCapacity * 2 / 3) {
				queue = Arrays.copyOf(queue, initialCapacity);
			} else {
				queue = Arrays.copyOf(queue, Math.max(size, (queue.length * 2) / 3));
			}
		}

		return result;
	}

	protected int peek(int location) {
		return queue[location];
	}

	public String toString() {
		return Arrays.toString(queue);
	}

	/**
	 * Inserts the specified element into this priority queue.
	 * 
	 * @return {@code true} (as specified by {@link Collection#add})
	 * @throws ClassCastException
	 *             if the specified element cannot be compared with elements
	 *             currently in this priority queue according to the priority
	 *             queue's ordering
	 * @throws NullPointerException
	 *             if the specified element is null
	 */
	public boolean add(int marking) {
		if (algorithm.getFScore(marking) > maxCost) {
			return false;
		}
		//		assert checkInv();
		// check if overwrite is necessary, i.e. only add if the object does not
		// exist yet,
		// or exists, but with higher costs.
		int location = locationMap.get(marking);
		if (location == NEV) {
			// new element, add to queue and return
			offer(marking);
			//			assert checkInv();
			return true;
		}

		// if the marking which exists at location has updated score
		// and the new score is better, then sift the marking up
		if (location > 0 && isBetter(marking, peek((location - 1) >>> 1))) {
			// update to better, if newE better then peek(location)
			siftUp(location, marking);
			//			assert checkInv();
			return true;
		} else if ((location << 1) + 1 < size && isBetter(peek((location << 1) + 1), marking)) {
			siftDown(location, marking);
			//			assert checkInv();
			return true;
		} else if ((location << 1) + 2 < size && isBetter(peek((location << 1) + 2), marking)) {
			siftDown(location, marking);
			//			assert checkInv();
			return true;
		}

		return false;
	}

	/**
	 * First order sorting is based on F score alone.
	 */
	protected boolean isBetter(int marking1, int marking2) {
		return algorithm.getFScore(marking1) < algorithm.getFScore(marking2);
	}

	/**
	 * Inserts the specified element into this priority queue.
	 * 
	 * @return {@code true} (as specified by {@link Queue#offer})
	 * @throws ClassCastException
	 *             if the specified element cannot be compared with elements
	 *             currently in this priority queue according to the priority
	 *             queue's ordering
	 * @throws NullPointerException
	 *             if the specified element is null
	 */
	protected void offer(int marking) {
		int i = size;
		if (i >= queue.length)
			grow(i + 1);
		size = i + 1;
		if (i == 0) {
			queue[0] = marking;
			locationMap.put(marking, 0);
		} else
			siftUp(i, marking);

		if (size > maxQueueSize) {
			maxQueueSize = size;
		}
	}

	/**
	 * Inserts item x at position k, maintaining heap invariant by promoting x up
	 * the tree until it is greater than or equal to its parent, or is the root.
	 * 
	 * @param fromPosition
	 * @param marking
	 *            the item to insert
	 */
	protected void siftUp(int fromPosition, int marking) {
		while (fromPosition > 0) {
			int parent = (fromPosition - 1) >>> 1;
			int existing = queue[parent];
			if (!isBetter(marking, existing)) {
				break;
			}
			queue[fromPosition] = existing;
			locationMap.put(existing, fromPosition);
			fromPosition = parent;
		}
		queue[fromPosition] = marking;
		locationMap.put(marking, fromPosition);
	}

	/**
	 * Inserts item x at position k, maintaining heap invariant by demoting x down
	 * the tree repeatedly until it is less than or equal to its children or is a
	 * leaf.
	 * 
	 * @param positionToFill
	 *            the position to fill
	 * @param marking
	 *            the item to insert
	 */
	protected void siftDown(int positionToFill, int marking) {
		int half = size >>> 1;
		while (positionToFill < half) {
			int child = (positionToFill << 1) + 1;
			int c = queue[child];
			int right = child + 1;
			if (right < size && isBetter(queue[right], c))
				c = queue[child = right];

			if (!isBetter(c, marking))
				break;

			queue[positionToFill] = c;
			// assert locationMap.get(c.getState()) == child;
			// i.e. child + k -child == k,
			// hence we use adjustValue instead of put here.
			locationMap.adjustValue(c, positionToFill - child);
			positionToFill = child;
		}
		queue[positionToFill] = marking;
		locationMap.put(marking, positionToFill);
	}

	protected boolean checkInv(int loc) {
		int n = queue[loc];
		int c1 = NEV;
		int c2 = NEV;
		if (2 * loc + 1 < size)
			c1 = queue[2 * loc + 1];

		if (2 * (loc + 1) < size)
			c2 = queue[2 * (loc + 1)];

		if (c1 != NEV) {
			if (isBetter(c1, n)) {
				System.err.println("Child " + c1 + "(" + algorithm.getFScore(c1) + ") is better than parent " + n + "("
						+ algorithm.getFScore(n) + ")");
				return false;
			}
		}
		if (c2 != NEV) {
			if (isBetter(c2, n)) {
				System.err.println("Child " + c2 + "(" + algorithm.getFScore(c2) + ") is better than parent " + n + "("
						+ algorithm.getFScore(n) + ")");
				return false;
			}
		}
		return (c1 == NEV ? true : checkInv(2 * loc + 1)) && (c2 == NEV ? true : checkInv(2 * (loc + 1)));

	}

	public int maxCapacity() {
		return maxQueueLength;
	}

	public int maxSize() {
		return maxQueueSize;
	}

	public long getEstimatedMemorySize() {
		return 4 + 4 * maxQueueLength + maxMapCapacity * 9 + 40;
	}
}