package nl.tue.alignment.algorithms.syncproduct;

import java.util.Arrays;

public class ObjectList<T> {

	private Object[] list;
	private int size = 0;

	public ObjectList(int capacity) {
		list = new Object[capacity];
	}

	public void add(T s) {
		ensureCapacity(size);
		list[size] = s;
		size++;
	}

	private void ensureCapacity(int insertAt) {
		if (list.length <= insertAt) {
			list = Arrays.copyOf(list, list.length * 2);
		}
	}

	public void truncate(int size) {
		this.size = size;
		Arrays.fill(list, size, list.length, null);
	}

	public T get(int index) {
		return (T) list[index];
	}

	public int size() {
		return size;
	}

	public T[] toArray(T[] a) {
		if (a.length < size)
			// Make a new array of a's runtime type, but my contents:
			return (T[]) Arrays.copyOf(list, size, a.getClass());
		System.arraycopy(list, 0, a, 0, size);
		if (a.length > size)
			a[size] = null;
		return a;
	}

	public String toString() {

		int iMax = size - 1;
		if (iMax == -1)
			return "[]";

		StringBuilder b = new StringBuilder();
		b.append('[');
		for (int i = 0;; i++) {
			b.append(list[i]);
			if (i == iMax)
				return b.append(']').toString();
			b.append(", ");
		}
	}

	public void set(int index, T value) {
		list[index] = value;
	}

}