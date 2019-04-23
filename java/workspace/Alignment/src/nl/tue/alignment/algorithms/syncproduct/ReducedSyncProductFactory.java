package nl.tue.alignment.algorithms.syncproduct;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Map;

import org.deckfour.xes.classification.XEventClass;
import org.deckfour.xes.classification.XEventClasses;
import org.deckfour.xes.extension.std.XConceptExtension;
import org.deckfour.xes.extension.std.XTimeExtension;
import org.deckfour.xes.model.XEvent;
import org.deckfour.xes.model.XTrace;
import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.graphbased.directed.petrinet.elements.Transition;
import org.processmining.models.semantics.petrinet.Marking;
import org.processmining.plugins.connectionfactories.logpetrinet.TransEvClassMapping;
import org.processmining.plugins.petrinet.replayresult.PNRepResult;
import org.processmining.plugins.petrinet.replayresult.StepTypes;
import org.processmining.plugins.replayer.replayresult.SyncReplayResult;

import gnu.trove.iterator.TObjectIntIterator;
import gnu.trove.list.TByteList;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TByteArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import gnu.trove.procedure.TObjectIntProcedure;
import nl.tue.alignment.Replayer;
import nl.tue.alignment.Utils;
import nl.tue.alignment.Utils.Statistic;
import nl.tue.alignment.algorithms.syncproduct.petrinet.ReducedPetriNet;
import nl.tue.alignment.algorithms.syncproduct.petrinet.ReducedPlace;
import nl.tue.alignment.algorithms.syncproduct.petrinet.ReducedTransition;
import nl.tue.alignment.algorithms.syncproduct.petrinet.TransitionEventClassList;
import nl.tue.astar.Trace;
import nl.tue.astar.util.LinearTrace;
import nl.tue.astar.util.PartiallyOrderedTrace;

public class ReducedSyncProductFactory implements SyncProductFactory<ReducedTransition> {

	private TObjectIntMap<XEventClass> c2id;
	private ReducedPetriNet reducedNet;
	private TObjectIntMap<Transition> trans2id;
	private XEventClasses classes;

	private final ObjectList<String> t2name;
	private final ObjectList<String> p2name;
	private final ObjectList<int[]> t2eid;
	private TIntList ranks = new TIntArrayList();
	private TByteList t2type = new TByteArrayList();
	private TIntList t2mmCost = new TIntArrayList();
	private int transitions;
	private int places;
	private int classCount;
	private int[] c2lmCost;
	private int maxSequenceLength;
	private final ObjectList<int[]> t2input;
	private final ObjectList<int[]> t2output;
	private TObjectIntMap<ReducedPlace> p2id;
	private final byte[] initMarking;
	private final byte[] finMarking;

	public ReducedSyncProductFactory(Petrinet net, XEventClasses classes, TObjectIntMap<XEventClass> c2id,
			TransEvClassMapping map, Marking initialMarking, Marking finalMarking, int maxSequenceLength) {
		this(net, classes, c2id, map, new GenericMap2Int<Transition>(1), new GenericMap2Int<XEventClass>(1), //
				new GenericMap2Int<Transition>(0), initialMarking, finalMarking, maxSequenceLength);
	}

	public ReducedSyncProductFactory(Petrinet net, XEventClasses classes, TObjectIntMap<XEventClass> c2id,
			TransEvClassMapping map, Map<Transition, Integer> mapTrans2Cost, Map<XEventClass, Integer> mapEvClass2Cost,
			Map<Transition, Integer> mapSync2Cost, Marking initialMarking, Marking finalMarking,
			int maxSequenceLength) {
		this(net, classes, c2id, map, new GenericMap2Int<>(mapTrans2Cost, 1), new GenericMap2Int<>(mapEvClass2Cost, 1), //
				new GenericMap2Int<>(mapSync2Cost, 0), initialMarking, finalMarking, maxSequenceLength);
	}

	public ReducedSyncProductFactory(Petrinet net, XEventClasses classes, TObjectIntMap<XEventClass> c2id,
			TransEvClassMapping map, TObjectIntMap<Transition> mapTrans2Cost,
			TObjectIntMap<XEventClass> mapEvClass2Cost, TObjectIntMap<Transition> mapSync2Cost, Marking initialMarking,
			Marking finalMarking, int maxSequenceLength) {
		this(net, classes, c2id, map, new GenericMap2Int<>(mapTrans2Cost, 1), new GenericMap2Int<>(mapEvClass2Cost, 1), //
				new GenericMap2Int<>(mapSync2Cost, 0), initialMarking, finalMarking, maxSequenceLength);
	}

	public ReducedSyncProductFactory(Petrinet net, XEventClasses classes, TObjectIntMap<XEventClass> c2id,
			TransEvClassMapping map, Map<Transition, Integer> mapTrans2Cost, Map<XEventClass, Integer> mapEvClass2Cost,
			Marking initialMarking, Marking finalMarking, int maxSequenceLength) {
		this(net, classes, c2id, map, new GenericMap2Int<>(mapTrans2Cost, 1), new GenericMap2Int<>(mapEvClass2Cost, 1), //
				new GenericMap2Int<Transition>(0), initialMarking, finalMarking, maxSequenceLength);
	}

	public ReducedSyncProductFactory(Petrinet net, XEventClasses classes, TObjectIntMap<XEventClass> c2id,
			TransEvClassMapping map, TObjectIntMap<Transition> mapTrans2Cost,
			TObjectIntMap<XEventClass> mapEvClass2Cost, Marking initialMarking, Marking finalMarking,
			int maxSequenceLength) {
		this(net, classes, c2id, map, new GenericMap2Int<>(mapTrans2Cost, 1), new GenericMap2Int<>(mapEvClass2Cost, 1), //
				new GenericMap2Int<Transition>(0), initialMarking, finalMarking, maxSequenceLength);
	}

	private ReducedSyncProductFactory(Petrinet net, XEventClasses classes, TObjectIntMap<XEventClass> c2id,
			TransEvClassMapping map, GenericMap2Int<Transition> mapTrans2Cost,
			GenericMap2Int<XEventClass> mapEvClass2Cost, GenericMap2Int<Transition> mapSync2Cost,
			Marking initialMarking, Marking finalMarking, int maxSequenceLength) {

		this.c2id = c2id;
		this.classes = classes;
		this.maxSequenceLength = maxSequenceLength;

		// find the highest class number
		int mx = 0;
		TObjectIntIterator<XEventClass> its = c2id.iterator();
		while (its.hasNext()) {
			its.advance();
			if (mx < its.value()) {
				mx = its.value();
			}
		}
		mx++;

		this.classCount = mx;
		c2lmCost = new int[classCount];
		for (XEventClass clazz : classes.getClasses()) {
			c2lmCost[c2id.get(clazz)] = mapEvClass2Cost.get(clazz);
		}

		// setup internal structures
		trans2id = new TObjectIntHashMap<>();
		int i = 0;
		for (Transition t : net.getTransitions()) {
			trans2id.put(t, i);
			i++;
		}

		// produce a Petri net for reduction
		reducedNet = new ReducedPetriNet(net, classes, trans2id, c2id, map, mapTrans2Cost, mapEvClass2Cost,
				mapSync2Cost, initialMarking, finalMarking);

		// start reducing the model into a new model applying as many rules as possible.
		// reduce the net to a minimum
		reducedNet.reduce(Integer.MAX_VALUE, maxSequenceLength);

		//		PrintStream writer;
		//		try {
		//			i = 0;
		//			int step = 1;
		//			do {
		//				writer = new PrintStream(new File(String.format("c://temp//dot//model%03d.dot", i)));
		//				reducedNet.toDot(writer);
		//				writer.close();
		//				i += step;
		//			} while (reducedNet.reduce(step, maxSequenceLength));
		//			writer = new PrintStream(new File(String.format("c://temp//dot//model%03d.dot", i)));
		//			reducedNet.toDot(writer);
		//			writer.close();
		//
		//		} catch (FileNotFoundException e) {
		//			// TODO Auto-generated catch block
		//			e.printStackTrace();
		//		}

		// prepare Data Structures for synchronous product.
		this.transitions = reducedNet.getTransitions().size();
		this.places = reducedNet.getPlaces().size();

		t2name = new ObjectList<>(transitions * 2);
		t2input = new ObjectList<>(transitions * 2);
		t2output = new ObjectList<>(transitions * 2);
		t2eid = new ObjectList<>(transitions * 2);
		ranks = new TIntArrayList();
		t2type = new TByteArrayList();
		t2mmCost = new TIntArrayList();

		p2name = new ObjectList<>(places * 2);

		// store a map from places to their IDs.
		p2id = new TObjectIntHashMap<>(net.getPlaces().size(), 0.75f, -1);
		int id = 0;
		for (ReducedPlace p : reducedNet.getPlaces()) {
			p2name.add(p.toIdString());
			p2id.put(p, id);
			id++;
		}

		for (ReducedTransition t : reducedNet.getTransitions()) {
			// transition label
			t2name.add(t.toIdString());
			t2eid.add(SyncProduct.NOEVENT);
			ranks.add(SyncProduct.NORANK);
			t2type.add(SyncProduct.MODEL_MOVE);
			t2mmCost.add(t.getModelMoveCost());

			final TIntList places = new TIntArrayList(this.places);
			t.forEachInputArc(new TObjectIntProcedure<ReducedPlace>() {

				public boolean execute(ReducedPlace a, int b) {
					int pid = p2id.get(a);
					for (int i = 0; i < b; i++) {
						places.add(pid);
					}
					return true;
				}
			});
			t2input.add(places.toArray());

			places.clear();
			t.forEachOutputArc(new TObjectIntProcedure<ReducedPlace>() {

				public boolean execute(ReducedPlace a, int b) {
					int pid = p2id.get(a);
					for (int i = 0; i < b; i++) {
						places.add(pid);
					}
					return true;
				}
			});
			t2output.add(places.toArray());

		}

		// mark the initial places
		initMarking = new byte[p2name.size()];
		reducedNet.getInitialMarking().forEachEntry(new TObjectIntProcedure<ReducedPlace>() {

			public boolean execute(ReducedPlace a, int b) {
				int id = p2id.get(a);
				if (b > 0 && id >= 0) {
					initMarking[id] = (byte) b;
				}
				return true;
			}
		});
		finMarking = new byte[p2name.size()];
		reducedNet.getFinalMarking().forEachEntry(new TObjectIntProcedure<ReducedPlace>() {

			public boolean execute(ReducedPlace a, int b) {
				int id = p2id.get(a);
				if (b > 0 && id >= 0) {
					finMarking[id] = (byte) b;
				}
				return true;
			}
		});

	}

	public synchronized SyncProduct getSyncProductForEmptyTrace(ArrayList<? super ReducedTransition> transitionList) {
		SyncProduct result = getLinearSyncProduct(new LinearTrace("Empty", 0), transitionList);
		return result;
	}

	public Trace getTrace(XTrace xTrace, boolean partiallyOrderSameTimestamp) {
		String traceLabel = XConceptExtension.instance().extractName(xTrace);
		if (traceLabel == null) {
			traceLabel = "XTrace@" + Integer.toHexString(xTrace.hashCode());
		}

		if (partiallyOrderSameTimestamp) {
			return getPartiallyOrderedTrace(xTrace, traceLabel);
		} else {
			return getLinearTrace(xTrace, traceLabel);
		}
	}

	public synchronized SyncProduct getSyncProduct(XTrace xTrace, ArrayList<? super ReducedTransition> transitionList,
			boolean partiallyOrderSameTimestamp) {
		String traceLabel = XConceptExtension.instance().extractName(xTrace);
		if (traceLabel == null) {
			traceLabel = "XTrace@" + Integer.toHexString(xTrace.hashCode());
		}
		if (partiallyOrderSameTimestamp) {
			PartiallyOrderedTrace trace = getPartiallyOrderedTrace(xTrace, traceLabel);
			// Do the ranking on this trace.
			throw new UnsupportedOperationException("Cannot handle partially ordered traces yet");
			//			return getPartiallyOrderedSyncProduct(trace, transitionList);
		} else {
			SyncProduct result = getLinearSyncProduct(getLinearTrace(xTrace, traceLabel), transitionList);
			return result;
		}

	}

	private SyncProduct getLinearSyncProduct(final LinearTrace trace,
			final ArrayList<? super ReducedTransition> transitionList) {
		transitionList.clear();

		//		// Compute the direct succession relation inside the trace
		//		int[][] succ = new int[classCount][classCount];
		//		for (int e2 = 1; e2 < trace.getSize(); e2++) {
		//			for (int e1 = Math.max(0, e2 - 3 * maxSequenceLength); e1 < e2; e1++) {
		//				succ[trace.get(e1)][trace.get(e2)]++;
		//			}
		//		}
		//
		//		// we sequence-reduce transitions <a,b>, only if succ[b][a] > 0
		//		// so only if we actually 
		
		// set the ranks of the model moves to NORANK
		final TIntList ranks = new TIntArrayList();
		final TIntList pathLengths = new TIntArrayList();
		for (int t = 0; t < transitions; t++) {
			ranks.add(SyncProduct.NORANK);
			pathLengths.add(1);
		}

		for (int e = 0; e < trace.getSize(); e++) {
			// add a place
			int cid = trace.get(e);
			p2name.add("e" + e);
			// add log move
			t2name.add("e" + e + " (" + cid + ")");//clazz.toString());
			t2mmCost.add(c2lmCost[cid]);
			t2eid.add(new int[] { e });
			t2type.add(SyncProduct.LOG_MOVE);
			// add input from just created place
			t2input.add(new int[] { places + e });
			// add output to future created place
			t2output.add(new int[] { places + e + 1 });
			ranks.add(e);
			transitionList.add(null);
			pathLengths.add(1);
		}
		if (trace.getSize() > 0) {
			p2name.add("e" + trace.getSize());
		}
		transitionList.addAll(reducedNet.getTransitions());

		for (int rti = reducedNet.getTransitions().size(); rti-- > 0;) {
			final ReducedTransition rt = reducedNet.getTransitions().get(rti);
			final int rtIndex = rti;
			rt.forEachSynchronousSequence(new TObjectIntProcedure<TransitionEventClassList>() {

				ReducedTransition[] split;

				private void match(TransitionEventClassList list, int placeToProduceIn, int cost, int seqIndex,
						int eventIndex) {
					if (eventIndex < 0 || seqIndex < 0 || seqIndex > eventIndex) {
						// base case, we're done.
						return;
					}

					if (trace.get(eventIndex) == list.getEventClassSequence()[seqIndex]) {
						// match so add transition to model
						t2name.add("t" + t2name.size() + "<br/>" + Arrays.toString(list.getEventClassSequence()) + "["
								+ seqIndex + "]");
						t2mmCost.add(seqIndex == 0 ? cost : 0);
						t2eid.add(new int[] { eventIndex });
						t2type.add(SyncProduct.SYNC_MOVE);
						transitionList.add(split[seqIndex]);
						ranks.add(eventIndex);
						pathLengths.add(1);

						if (seqIndex == list.getEventClassSequence().length - 1) {
							// this is the final transition in the sequence
							// add output equal to the output of the net.
							int[] output = Arrays.copyOf(t2output.get(rtIndex), t2output.get(rtIndex).length + 1);
							// and to the event output place in the trace
							output[output.length - 1] = places + eventIndex + 1;
							t2output.add(output);
						} else {
							// output is simply the place in the sequence,
							// plus the trace
							assert placeToProduceIn > places + trace.getSize();
							t2output.add(new int[] { placeToProduceIn, places + eventIndex + 1 });
						}
						if (seqIndex == 0) {
							// this is the first transition in the sequence.
							// add input from just created place on the log side
							// and all inputs from the model side
							int[] input = Arrays.copyOf(t2input.get(rtIndex), t2input.get(rtIndex).length + 1);
							input[input.length - 1] = places + eventIndex;
							t2input.add(input);
						} else {
							// add an input place
							int newPlaceToProduceIn = p2name.size();
							p2name.add("i_" + (t2name.size() - 1) + "-" + eventIndex);
							t2input.add(new int[] { newPlaceToProduceIn, places + eventIndex });

							// and continue with predecessor either
							// by matching predecessor events
							match(list, newPlaceToProduceIn, cost, seqIndex - 1, eventIndex - 1);
						}

					}
					match(list, placeToProduceIn, cost, seqIndex, eventIndex - 1);
				}

				public boolean execute(TransitionEventClassList list, int cost) {
					int e = trace.getSize() - 1;
					while (e >= list.getEventClassSequence().length && !list.endsWith(trace.get(e))) {
						e--;
					}
					if (e >= 0 && e + 1 >= list.getEventClassSequence().length) {
						split = ReducedTransition.createList(list, cost, rt.getModelMoveCost());

						match(list, -1, cost, list.getEventClassSequence().length - 1, e);
					}
					return true;
				}
			});

		}

		//			// now find sync moves.
		//			TIntList seqList = new TIntArrayList(maxSequenceLength);
		//			TIntList evtList = new TIntArrayList(maxSequenceLength);
		//			int i = 0;
		//			do {
		//				evtList.insert(0, e - i);
		//				seqList.insert(0, trace.get(e - i));
		//				int[] evt = evtList.toArray();
		//				int[] seq = seqList.toArray();
		//				// look at the sequences starting with event e and find potential 
		//				// synchronous transitions matching this sequence.
		//				int ti = 0;
		//				for (ReducedTransition rt : reducedNet.getTransitions()) {
		//					if (rt.mapsTo(seq)) {
		//						// transition rt can be mapped to a sequence seq.
		//						t2name.add(t2name.get(ti) + " + e" + Arrays.toString(evt) + "(" + Arrays.toString(seq) + ")");
		//						t2mmCost.add(rt.getCostFor(seq));
		//						t2eid.add(evt);
		//						t2type.add(SyncProduct.SYNC_MOVE);
		//						ranks.add(e);
		//					}
		//					ti++;
		//				}
		//				i++;
		//			} while (i < maxSequenceLength && e - i >= 0);

		//		}

		SyncProductImpl product = new SyncProductImpl(trace.getLabel(), //label
				this.classCount, // number of classes
				t2name.toArray(new String[t2name.size()]), //transition labels
				p2name.toArray(new String[p2name.size()]), // place labels
				t2eid.toArray(new int[t2eid.size()][]), //event numbers
				ranks.toArray(), // ranks
				pathLengths.toArray(), // pathlengths
				t2type.toArray(), //types
				t2mmCost.toArray());

		int t = 0;
		for (; t < t2input.size(); t++) {
			// first the model moves
			product.setInput(t, t2input.get(t));
			product.setOutput(t, t2output.get(t));
		}

		product.setInitialMarking(Arrays.copyOf(initMarking, p2name.size()));
		product.setFinalMarking(Arrays.copyOf(finMarking, p2name.size()));
		if (trace.getSize() > 0) {
			product.addToInitialMarking(places);
			product.addToFinalMarking(places + trace.getSize());
		}

		// trim to size;
		p2name.truncate(places);
		t2name.truncate(transitions);
		t2mmCost.remove(transitions, t2mmCost.size() - transitions);
		t2eid.truncate(transitions);//, t2eid.size() - transitions);
		t2type.remove(transitions, t2type.size() - transitions);
		t2input.truncate(transitions);
		t2output.truncate(transitions);
		return product;
	}

	private LinearTrace getLinearTrace(XTrace xTrace, String label) {
		LinearTrace trace = new LinearTrace(label, xTrace.size());
		for (int e = 0; e < xTrace.size(); e++) {
			XEventClass clazz = classes.getClassOf(xTrace.get(e));
			trace.set(e, c2id.get(clazz));
		}

		return trace;

	}

	//	private SyncProduct getPartiallyOrderedSyncProduct(PartiallyOrderedTrace trace, List<Transition> transitionList) {
	//		transitionList.clear();
	//
	//		int[] e2t = new int[trace.getSize()];
	//
	//		// for this trace, compute the log-moves
	//		// compute the sync moves
	//		for (int e = 0; e < trace.getSize(); e++) {
	//			//			XEventClass clazz = classes.getClassOf(trace.get(e));
	//			int cid = trace.get(e); // c2id.get(clazz);
	//			int[] predecessors = trace.getPredecessors(e);
	//			if (predecessors == null) {
	//				// initial place
	//				// add a place
	//				p2name.add("p_init-" + e);
	//			} else {
	//				for (int pi = 0; pi < predecessors.length; pi++) {
	//					// add a place
	//					p2name.add("p_" + predecessors[pi] + "-" + e);
	//				}
	//			}
	//
	//			e2t[e] = t2name.size();
	//			// add log move
	//			t2name.add("e" + e + "(" + cid + ")");//clazz.toString());
	//			t2mmCost.add(c2lmCost[cid]);
	//			t2eid.add(new int[] { e });
	//			t2type.add(SyncProduct.LOG_MOVE);
	//
	//			TIntSet set = c2t.get(cid);
	//			if (set != null) {
	//				TIntIterator it = set.iterator();
	//				while (it.hasNext()) {
	//					// add sync move
	//					int t = it.next();
	//					t2name.add(t2name.get(t) + ",e" + e + "(" + cid + ")");
	//					t2mmCost.add(t2smCost.get(t));
	//					t2eid.add(new int[] { e });
	//					t2type.add(SyncProduct.SYNC_MOVE);
	//				}
	//			}
	//
	//		}
	//
	//		int[] ranks = new int[t2eid.size()];
	//		Arrays.fill(ranks, SyncProduct.NORANK);
	//		SyncProductImpl product = new SyncProductImpl(trace.getLabel(), //label
	//				this.classCount, //number of event classes
	//				t2name.toArray(new String[t2name.size()]), //transition labels
	//				p2name.toArray(new String[p2name.size()]), // place labels
	//				t2eid.toArray(new int[t2eid.size()][]), //event numbers
	//				ranks, // ranks
	//				t2type.toArray(), //types
	//				t2mmCost.toArray());
	//
	//		int t = 0;
	//		for (; t < transitions; t++) {
	//			// first the model moves
	//			product.setInput(t, t2input.get(t));
	//			product.setOutput(t, t2output.get(t));
	//			transitionList.add(t2transition[t]);
	//		}
	//
	//		product.setInitialMarking(Arrays.copyOf(initMarking, p2name.size()));
	//		product.setFinalMarking(Arrays.copyOf(finMarking, p2name.size()));
	//
	//		// TODO: Handle the sync product ranking properly. Currently, a random sequence
	//		// of events is ranked and the assumption is that events are ordered, i.e. that 
	//		// the predecessors of the event at index e are a index < e .
	//
	//		int minRank = SyncProduct.NORANK;
	//		int p = places;
	//		for (int e = 0; e < trace.getSize(); e++) {
	//			int cid = trace.get(e); // c2id.get(clazz);
	//
	//			int[] predecessors = trace.getPredecessors(e);
	//			if (predecessors == null) {
	//				// initial place
	//				// add a place
	//				if (minRank == SyncProduct.NORANK) {
	//					product.setRankOf(e2t[e], ++minRank);
	//				}
	//				product.setInput(e2t[e], p);
	//				product.addToInitialMarking(p);
	//				p++;
	//			} else {
	//				for (int pi = 0; pi < predecessors.length; pi++) {
	//					// add a place
	//					if (product.getRankOf(e2t[predecessors[pi]]) == minRank) {
	//						product.setRankOf(e2t[e], ++minRank);
	//					}
	//					product.addToInput(e2t[e], p);
	//					product.addToOutput(e2t[predecessors[pi]], p);
	//					p++;
	//				}
	//
	//			}
	//			transitionList.add(null);
	//
	//			TIntSet set = c2t.get(cid);
	//			if (set != null) {
	//				TIntIterator it = set.iterator();
	//				while (it.hasNext()) {
	//					// add sync move
	//					int t2 = it.next();
	//
	//					transitionList.add(t2transition[t2]);
	//				}
	//			}
	//		}
	//		for (int e = 0; e < trace.getSize(); e++) {
	//			int cid = trace.get(e); // c2id.get(clazz);
	//			t++;
	//			TIntSet set = c2t.get(cid);
	//			if (set != null) {
	//				TIntIterator it = set.iterator();
	//				while (it.hasNext()) {
	//					// add sync move
	//					int t2 = it.next();
	//					product.setInput(t, t2input.get(t2));
	//					product.setOutput(t, t2output.get(t2));
	//
	//					product.addToInput(t, product.getInput(e2t[e]));
	//					product.addToOutput(t, product.getOutput(e2t[e]));
	//
	//					t++;
	//				}
	//			}
	//		}
	//
	//		// trim to size;
	//		p2name.trunctate(places);
	//		t2name.trunctate(transitions);
	//		t2mmCost.remove(transitions, t2mmCost.size() - transitions);
	//		//		t2eid.remove(transitions, t2eid.size() - transitions);
	//		t2eid = t2eid.subList(0, transitions);
	//		t2type.remove(transitions, t2type.size() - transitions);
	//
	//		return product;
	//	}

	private PartiallyOrderedTrace getPartiallyOrderedTrace(XTrace xTrace, String label) {
		int s = xTrace.size();
		int[] idx = new int[s];

		TIntList activities = new TIntArrayList(s);
		List<int[]> predecessors = new ArrayList<int[]>();
		Date lastTime = null;
		TIntList pre = new TIntArrayList();
		int previousIndex = -1;
		int currentIdx = 0;
		for (int i = 0; i < s; i++) {
			XEvent event = xTrace.get(i);
			int act = c2id.get(classes.getClassOf(event));
			//			int act = delegate.getActivityOf(trace, i);
			idx[i] = currentIdx;
			Date timestamp = XTimeExtension.instance().extractTimestamp(event);

			activities.add(act);

			if (lastTime == null) {
				// first event
				predecessors.add(null);
			} else if (timestamp.equals(lastTime)) {
				// timestamp is the same as the last event.
				if (previousIndex >= 0) {
					predecessors.add(new int[] { previousIndex });
				} else {
					predecessors.add(null);
				}
			} else {
				// timestamp is different from the last event.
				predecessors.add(pre.toArray());
				previousIndex = idx[i - 1];
				pre = new TIntArrayList();
			}
			pre.add(currentIdx);
			lastTime = timestamp;
			currentIdx++;

		}

		PartiallyOrderedTrace result;
		// predecessors[i] holds all predecessors of event at index i
		result = new PartiallyOrderedTrace(label, activities.toArray(), predecessors.toArray(new int[0][]));
		return result;

	}

	public SyncReplayResult toSyncReplayResult(Replayer replayer, SyncProduct product,
			TObjectIntMap<Statistic> statistics, int[] alignment, XTrace trace, int traceIndex,
			ArrayList<? super ReducedTransition> transitionList) {
		List<Object> nodeInstance = new ArrayList<>(alignment.length);
		List<StepTypes> stepTypes = new ArrayList<>(alignment.length);
		int mm = 0, lm = 0, smm = 0, slm = 0;
		for (int i = 0; i < alignment.length; i++) {
			int t = alignment[i];
			if (product.getTypeOf(t) == SyncProduct.LOG_MOVE) {
				int[] events = product.getEventOf(t);
				for (int e : events) {
					// a log move is a list of events. Most likely just one.
					nodeInstance.add(classes.getClassOf(trace.get(e)));
					stepTypes.add(StepTypes.L);
					lm += product.getCost(t);
				}
			} else {
				nodeInstance.add(transitionList.get(t));
				if (product.getTypeOf(t) == SyncProduct.MODEL_MOVE) {
					stepTypes.add(StepTypes.MREAL);
					mm += product.getCost(t);
				} else if (product.getTypeOf(t) == SyncProduct.SYNC_MOVE) {
					int[] events = product.getEventOf(t);
					smm += ((ReducedTransition) transitionList.get(t)).getModelMoveCost();
					for (int e : events) {
						stepTypes.add(StepTypes.LMGOOD);
						slm += replayer.getCostLM(classes.getClassOf(trace.get(e)));
					}
				} else if (product.getTypeOf(t) == SyncProduct.TAU_MOVE) {
					stepTypes.add(StepTypes.MINVI);
					mm += product.getCost(t);
				}
			}
		}

		SyncReplayResult srr = new SyncReplayResult(nodeInstance, stepTypes, traceIndex);
		srr.addInfo(PNRepResult.RAWFITNESSCOST, 1.0 * statistics.get(Statistic.COST));
		srr.addInfo(PNRepResult.TIME, (statistics.get(Statistic.TOTALTIME)) / 1000.0);
		srr.addInfo(PNRepResult.QUEUEDSTATE, 1.0 * statistics.get(Statistic.QUEUEACTIONS));
		if (lm + slm == 0) {
			srr.addInfo(PNRepResult.MOVELOGFITNESS, 1.0);
		} else {
			srr.addInfo(PNRepResult.MOVELOGFITNESS, 1.0 - (1.0 * lm) / (lm + slm));
		}
		if (mm + smm == 0) {
			srr.addInfo(PNRepResult.MOVEMODELFITNESS, 1.0);
		} else {
			srr.addInfo(PNRepResult.MOVEMODELFITNESS, 1.0 - (1.0 * mm) / (mm + smm));
		}
		srr.addInfo(PNRepResult.NUMSTATEGENERATED, 1.0 * statistics.get(Statistic.MARKINGSREACHED));
		srr.addInfo(PNRepResult.ORIGTRACELENGTH, 1.0 * trace.size());
		srr.addInfo(Replayer.TRACEEXITCODE, new Double(statistics.get(Statistic.EXITCODE)));
		srr.addInfo(Replayer.MEMORYUSED, new Double(statistics.get(Statistic.MEMORYUSED)));
		srr.addInfo(Replayer.PREPROCESSTIME, (statistics.get(Statistic.PREPROCESSTIME)) / 1000.0);
		srr.addInfo(Replayer.HEURISTICSCOMPUTED, (double) statistics.get(Statistic.HEURISTICSCOMPUTED));
		srr.setReliable(statistics.get(Statistic.EXITCODE) == Utils.OPTIMALALIGNMENT);
		return srr;
	}

	public SyncProduct getSyncProduct(XTrace xTrace, ArrayList<? super ReducedTransition> transitionList,
			boolean partiallyOrderSameTimestamp, boolean isPrefix) {
		// TODO Auto-generated method stub
		return null;
	}

}
