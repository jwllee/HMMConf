package nl.tue.alignment.algorithms.syncproduct;

import java.util.Map;

import gnu.trove.map.TObjectIntMap;

/**
 * wrapper around Map<K, Integer> and TObjectIntMap<K> to provide easy access to
 * the get(K key) method
 * 
 * @author bfvdonge
 *
 * @param <K>
 */
public class GenericMap2Int<K> {
	private final Map<K, Integer> map1;
	private final TObjectIntMap<K> map2;
	private final int defaultValue;

	private GenericMap2Int(Map<K, Integer> map1, TObjectIntMap<K> map2, int defaultValue) {
		this.map1 = map1;
		this.map2 = map2;
		this.defaultValue = defaultValue;
	}

	public GenericMap2Int(Map<K, Integer> map1, int defaultValue) {
		this(map1, null, defaultValue);
	}

	public GenericMap2Int(TObjectIntMap<K> map2, int defaultValue) {
		this(null, map2, defaultValue);
	}

	public GenericMap2Int(int defaultValue) {
		this(null, null, defaultValue);
	}

	public int get(K key) {
		if (map1 != null && map1.containsKey(key)) {
			return map1.get(key);
		} else if (map2 != null && map2.containsKey(key)) {
			return map2.get(key);
		} else {
			return defaultValue;
		}
	}
}