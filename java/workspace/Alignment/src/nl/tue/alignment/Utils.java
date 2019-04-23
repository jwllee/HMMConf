package nl.tue.alignment;

import java.io.PrintStream;

import org.deckfour.xes.classification.XEventClass;
import org.deckfour.xes.classification.XEventClasses;

import gnu.trove.iterator.TIntIntIterator;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import nl.tue.alignment.algorithms.syncproduct.SyncProduct;

public class Utils {

	public static final String SEP = ",";
	public static int OPTIMALALIGNMENT = 1;
	public static int FAILEDALIGNMENT = 2;
	public static int ENABLINGBLOCKEDBYOUTPUT = 4;
	public static int COSTFUNCTIONOVERFLOW = 8;
	public static int HEURISTICFUNCTIONOVERFLOW = 16;
	public static int TIMEOUTREACHED = 32;
	public static int STATELIMITREACHED = 64;
	public static int COSTLIMITREACHED = 128;
	public static int CANCELLED = 256;
	public static int FINALMARKINGUNREACHABLE = 512;

	/**
	 * Default block size determines how many bytes are reserved top store markings.
	 * Whenever a block is full, a new block of this size is allocated.
	 */
	public static int DEFAULTBLOCKSIZE = 1024;

	/**
	 * Initial size of the priority queue. It grows as needed.
	 */
	public static int DEFAULTQUEUESIZE = 16;

	/**
	 * Initial size of the visited state set. It grows as needed.
	 */
	public static int DEFAULTVISITEDSIZE = 16;

	public enum Statistic {
		EXITCODE("Exit code for alignment"), //
		ALIGNMENTLENGTH("Length of the alignment found"), //
		TRACELENGTH("Length of the orignal trace"), //
		PLACES("Places in the synchronous product"), //
		TRANSITIONS("Transtions in the synchronous product"), //
		COST("Cost of the alignment"), //
		EDGESTRAVERSED("Transitions fired"), //
		POLLACTIONS("Markings polled from queue"), //
		CLOSEDACTIONS("Markings added to closed set"), //
		QUEUEACTIONS("Markings queued"), //
		MARKINGSREACHED("Markings reached"), //
		HEURISTICSCOMPUTED("Heuristics computed"), //
		HEURISTICSESTIMATED("Heuristics estimated"), //
		HEURISTICSDERIVED("Heuristics derived"), //
		MAXQUEUELENGTH("Maximum queue length (elts)"), //
		MAXQUEUECAPACITY("Maximum queue capacity (elts)"), //
		VISITEDSETCAPACITY("Maximum capacity visited set (elts)"), //
		MEMORYUSED("Approximate peak memory used (kb)"), //
		RUNTIME("Time to compute alignment (us)"), //
		HEURISTICTIME("Time to compute heuristics (us)"), //
		SETUPTIME("Time to setup algorithm (us)"), //
		TOTALTIME("Total Time including setup (us)"), //
		SPLITS("Number of splits when splitting marking"), //
		LMCOST("Log move cost of alignment"), //
		MMCOST("Model move cost of alignment"), //
		SMCOST("Synchronous move cost of alignment"), //
		PREPROCESSTIME("Pre-processing time (us)"), //
		CONSTRAINTSETSIZE("Size of the constraintset"), //
		RESTARTS("Number of times replay was restarted");

		private final String label;

		private Statistic(String label) {
			this.label = label;
		}

		public String toString() {
			return label;
		}
	}

	public static String asVector(byte[] marking, SyncProduct net) {
		StringBuffer buf = new StringBuffer();
		buf.append('[');
		for (int i = 0; i < net.numPlaces();) {
			buf.append(marking[i]);
			if (++i < net.numPlaces()) {
				buf.append(',');
			}
		}
		buf.append(']');
		return buf.toString();
	}

	public static String asBag(byte[] marking, SyncProduct net) {
		StringBuffer buf = new StringBuffer();
		buf.append('[');
		for (int i = 0; i < net.numPlaces();) {
			if (marking[i] > 0) {
				if (buf.length() > 1) {
					buf.append(',');
				}
				if (marking[i] > 1) {
					buf.append(marking[i]);
				}
				buf.append(net.getPlaceLabel(i));
			}
			i++;
		}
		buf.append(']');
		return buf.toString();
	}

	public static void toTpn(SyncProduct product, PrintStream stream) {
		synchronized (stream) {
			for (int p = 0; p < product.numPlaces(); p++) {
				stream.print("place \"place_" + p);
				stream.print("\"");
				if (product.getInitialMarking()[p] > 0) {
					stream.print("init " + product.getInitialMarking()[p]);
				}
				stream.print(";\n");
			}
			for (int t = 0; t < product.numTransitions(); t++) {
				stream.print("trans \"t_" + t);
				stream.print("\"~\"");
				stream.print(
						product.getTypeOf(t) != SyncProduct.TAU_MOVE ? product.getTransitionLabel(t) : "invisible");
				stream.print("\" in ");
				for (int p : product.getInput(t)) {
					stream.print(" \"place_" + p);
					stream.print("\"");
				}
				stream.print(" out ");
				for (int p : product.getOutput(t)) {
					stream.print(" \"place_" + p);
					stream.print("\"");
				}
				stream.print(";\n");
			}
			stream.flush();
		}

	}

	public static void toTpnSplitStartComplete(SyncProduct product, PrintStream stream) {
		synchronized (stream) {
			for (int p = 0; p < product.numPlaces(); p++) {
				stream.print("place \"place_" + p);
				stream.print("\"");
				if (product.getInitialMarking()[p] > 0) {
					stream.print("init " + product.getInitialMarking()[p]);
				}
				stream.print(";\n");
			}
			for (int t = 0; t < product.numTransitions(); t++) {
				if (product.getTypeOf(t) != SyncProduct.TAU_MOVE) {
					stream.print("place \"internal_" + t);
					stream.print("\";\n");
				}
			}
			for (int t = 0; t < product.numTransitions(); t++) {
				stream.print("trans \"t_" + 2 * t);
				stream.print("\"");
				stream.print("~\"");
				stream.print(product.getTypeOf(t) != SyncProduct.TAU_MOVE ? product.getTransitionLabel(t) + "+start"
						: "invisible");
				stream.print("\"");
				stream.print(" in ");
				for (int p : product.getInput(t)) {
					stream.print(" \"place_" + p);
					stream.print("\"");
				}
				if (product.getTypeOf(t) != SyncProduct.TAU_MOVE) {
					stream.print(" out ");
					stream.print(" \"internal_" + t);
					stream.print("\"");
					stream.print(";\n");

					stream.print("trans \"t_" + (2 * t + 1));
					stream.print("\"~\"");
					stream.print(product.getTransitionLabel(t));
					stream.print("+complete");
					stream.print("\" in ");
					stream.print(" \"internal_" + t);
					stream.print("\"");
				}
				stream.print(" out ");
				for (int p : product.getOutput(t)) {
					stream.print(" \"place_" + p);
					stream.print("\"");
				}
				stream.print(";\n");

			}
			stream.flush();
		}

	}

	public static void toDot(SyncProduct product, PrintStream stream) {
		synchronized (stream) {
			stream.println("Digraph SP { \n rankdir=LR;\n");

			for (int p = 0; p < product.numPlaces(); p++) {
				placeToDot(product, stream, p, p);
			}
			stream.print("{ rank=same;");
			for (int p = 0; p < product.numPlaces(); p++) {
				if (product.getInitialMarking()[p] > 0) {
					stream.print("p" + p + "; ");
				}
			}
			stream.print("}\n");
			stream.print("{ rank=same;");
			for (int p = 0; p < product.numPlaces(); p++) {
				if (product.getFinalMarking()[p] > 0) {
					stream.print("p" + p + "; ");
				}
			}
			stream.print("}\n");

			TIntIntIterator it;
			//			TIntSet events = new TIntHashSet(product.numTransitions(), 0.5f, -3);
			for (int t = 0; t < product.numTransitions(); t++) {
				//				events.add(product.getEventOf(t));
				transitionToDot(product, stream, t, t);

				it = toBag(product.getInput(t)).iterator();
				while (it.hasNext()) {
					//				for (int p : product.getInput(t)) {
					it.advance();
					stream.print("p" + it.key() + " -> t" + t);
					if (product.getTypeOf(t) == SyncProduct.SYNC_MOVE) {
						stream.print(" [weight=2");
					} else {
						stream.print(" [weight=10");
					}
					if (it.value() > 1) {
						stream.print(",label=\"" + it.value() + "\"");
					}
					stream.print("];\n");
				}
				it = toBag(product.getOutput(t)).iterator();
				//				for (int p : product.getOutput(t)) {
				while (it.hasNext()) {
					it.advance();
					stream.print("t" + t + " -> p" + it.key());
					if (product.getTypeOf(t) == SyncProduct.SYNC_MOVE) {
						stream.print(" [weight=2");
					} else {
						stream.print(" [weight=10");
					}
					if (it.value() > 1) {
						stream.print(",label=\"" + it.value() + "\"");
					}
					stream.print("];\n");
				}
			}

			//			events.remove(SyncProduct.NOEVENT);

			//			int e;
			//			for (TIntIterator it2 = events.iterator(); it2.hasNext();) {
			//				e = it2.next();
			//				stream.print("{ rank=same;");
			//				for (int t = 0; t < product.numTransitions(); t++) {
			//					if (product.getEventOf(t) == e) {
			//						stream.print("t" + t + "; ");
			//					}
			//				}
			//				stream.print("}\n");
			//
			//			}

			stream.print("}");
			stream.flush();
		}

	}

	private static TIntIntMap toBag(int[] list) {
		TIntIntHashMap map = new TIntIntHashMap(list.length);
		for (int i : list) {
			map.adjustOrPutValue(i, 1, 1);
		}
		return map;
	}

	public static String toVector(SyncProduct product, double[] solution, int[] indexMap) {
		StringBuilder b = new StringBuilder();
		b.append("[");
		for (int i = 0; i < indexMap.length; i++) {
			if (solution[i] > 0) {
				b.append(String.format("%1$.2f ", solution[i]));
				b.append(product.getTransitionLabel(indexMap[i]));
				b.append("(");
				b.append(indexMap[i]);
				b.append(")");
				if (i < indexMap.length - 1) {
					b.append(", ");
				}
			}
		}
		b.append("]");
		return b.toString();
	}

	public static void toDot(SyncProduct product, int[] alignment, PrintStream stream) {
		synchronized (stream) {
			stream.print("Digraph A { \n rankdir=LR;\n");

			stream.print("{ rank=same;");
			TIntList[] place2index = new TIntList[product.numPlaces()];
			for (int p = 0; p < product.numPlaces(); p++) {
				place2index[p] = new TIntArrayList(3);
				if (product.getInitialMarking()[p] > 0) {
					place2index[p].add(p);
					stream.print("p" + p + "; ");
				}
			}
			stream.print("}\n");

			for (int i = 0; i < alignment.length; i++) {
				int t = alignment[i];
				transitionToDot(product, stream, i, t);

				for (int p : product.getInput(t)) {
					int j = place2index[p].size() > 0 ? place2index[p].removeAt(place2index[p].size() - 1) : -1;
					if (j == p) {
						placeToDot(product, stream, j, p);
					}
					stream.print("p" + j + " -> t" + i);
					if (product.getTypeOf(t) == SyncProduct.SYNC_MOVE) {
						stream.print(" [weight=2]");
					} else {
						stream.print(" [weight=10]");
					}
					stream.print(";\n");
				}
				for (int p : product.getOutput(t)) {
					int j = i * product.numPlaces() + p;
					place2index[p].add(j);
					placeToDot(product, stream, j, p);
					stream.print("t" + i + " -> p" + j);
					if (product.getTypeOf(t) == SyncProduct.SYNC_MOVE) {
						stream.print(" [weight=2]");
					} else {
						stream.print(" [weight=10]");
					}
					stream.print(";\n");
				}
			}

			stream.print("{ rank=same;");
			for (int p = 0; p < product.numPlaces(); p++) {
				if (place2index[p].size() > 0) {
					int i = place2index[p].get(0);
					stream.print("p" + i + "; ");
				}
			}
			stream.print("}\n");

			stream.print("}");
			stream.flush();
		}
	}

	private static void placeToDot(SyncProduct product, PrintStream stream, int i, int p) {
		stream.print("p" + i);
		stream.print(" [label=<" + product.getPlaceLabel(p));
		if (product.getInitialMarking()[p] > 0) {
			stream.print("<br/>i:" + product.getInitialMarking()[p]);
		}
		if (product.getFinalMarking()[p] > 0) {
			stream.print("<br/>f:" + product.getFinalMarking()[p]);
		}
		stream.print(">,shape=circle];");
		stream.print("\n");
	}

	private static void transitionToDot(SyncProduct product, PrintStream stream, int i, int t) {
		stream.print("t" + i);
		stream.print(" [label=<" + product.getTransitionLabel(t));
		stream.print("<br/>r:" + product.getRankOf(t));
		stream.print("<br/>c:" + product.getCost(t));
		stream.print(">");

		if (product.getTypeOf(t) == SyncProduct.LOG_MOVE) {
			stream.print(",style=filled,fillcolor=goldenrod2,fontcolor=black");
		} else if (product.getTypeOf(t) == SyncProduct.MODEL_MOVE) {
			stream.print(",style=filled,fillcolor=darkorchid1,fontcolor=white");
		} else if (product.getTypeOf(t) == SyncProduct.SYNC_MOVE) {
			stream.print(",style=filled,fillcolor=forestgreen,fontcolor=white");
		} else if (product.getTypeOf(t) == SyncProduct.TAU_MOVE) {
			stream.print(",style=filled,fillcolor=honeydew4,fontcolor=white");
		}

		stream.print(",shape=box];");
		stream.print("\n");
	}

	public static TObjectIntMap<XEventClass> createClass2ID(XEventClasses classes) {
		TObjectIntHashMap<XEventClass> c2id = new TObjectIntHashMap<>(classes.size(), 0.75f, -1);
		int id = 0;
		for (XEventClass clazz : classes.getClasses()) {
			c2id.put(clazz, id++);
		}
		return c2id;
	}

}
