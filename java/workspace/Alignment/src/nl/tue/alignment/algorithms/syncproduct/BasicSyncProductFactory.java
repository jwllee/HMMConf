package nl.tue.alignment.algorithms.syncproduct;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.deckfour.xes.classification.XEventClass;
import org.deckfour.xes.classification.XEventClasses;
import org.deckfour.xes.extension.std.XConceptExtension;
import org.deckfour.xes.extension.std.XTimeExtension;
import org.deckfour.xes.model.XEvent;
import org.deckfour.xes.model.XTrace;
import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.graphbased.directed.petrinet.PetrinetEdge;
import org.processmining.models.graphbased.directed.petrinet.elements.Arc;
import org.processmining.models.graphbased.directed.petrinet.elements.Place;
import org.processmining.models.graphbased.directed.petrinet.elements.Transition;
import org.processmining.models.semantics.petrinet.Marking;
import org.processmining.plugins.connectionfactories.logpetrinet.TransEvClassMapping;
import org.processmining.plugins.petrinet.replayresult.PNRepResult;
import org.processmining.plugins.petrinet.replayresult.StepTypes;
import org.processmining.plugins.replayer.replayresult.SyncReplayResult;

import gnu.trove.iterator.TIntIterator;
import gnu.trove.iterator.TObjectIntIterator;
import gnu.trove.list.TByteList;
import gnu.trove.list.TIntList;
import gnu.trove.list.array.TByteArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import gnu.trove.set.TIntSet;
import gnu.trove.set.hash.TIntHashSet;
import nl.tue.alignment.Replayer;
import nl.tue.alignment.Utils;
import nl.tue.alignment.Utils.Statistic;
import nl.tue.astar.Trace;
import nl.tue.astar.util.LinearTrace;
import nl.tue.astar.util.PartiallyOrderedTrace;

public class BasicSyncProductFactory implements SyncProductFactory<Transition> {

	private final int transitions;

	// transition to model moves
	private final TIntList t2mmCost;

	// eventClassSequence 2 sync move cost
	private final TIntList t2smCost;

	// maps transitions to IDs
	//	private final TObjectIntMap<Transition> t2id;
	private final ObjectList<String> t2name;
	private final ObjectList<int[]> t2input;
	private final ObjectList<int[]> t2output;
	private final ObjectList<int[]> t2eid;
	private final TByteList t2type;
	private final Transition[] t2transition;

	private final int classCount;
	private final int[] c2lmCost;
	// maps classes to sets of transitions representing these classes
	private final TIntObjectMap<TIntSet> c2t;

	private final int places;
	private final ObjectList<String> p2name;
	private final byte[] initMarking;
	private final byte[] finMarking;
	private final XEventClasses classes;
	private final TObjectIntMap<XEventClass> c2id;

	public BasicSyncProductFactory(Petrinet net, XEventClasses classes, TObjectIntMap<XEventClass> c2id,
			TransEvClassMapping map, Marking initialMarking, Marking finalMarking) {
		this(net, classes, c2id, map, new GenericMap2Int<Transition>(1), new GenericMap2Int<XEventClass>(1), //
				new GenericMap2Int<Transition>(0), initialMarking, finalMarking);
	}

	public BasicSyncProductFactory(Petrinet net, XEventClasses classes, TObjectIntMap<XEventClass> c2id,
			TransEvClassMapping map, Map<Transition, Integer> mapTrans2Cost, Map<XEventClass, Integer> mapEvClass2Cost,
			Map<Transition, Integer> mapSync2Cost, Marking initialMarking, Marking finalMarking) {
		this(net, classes, c2id, map, new GenericMap2Int<>(mapTrans2Cost, 1), new GenericMap2Int<>(mapEvClass2Cost, 1), //
				new GenericMap2Int<>(mapSync2Cost, 0), initialMarking, finalMarking);
	}

	public BasicSyncProductFactory(Petrinet net, XEventClasses classes, TObjectIntMap<XEventClass> c2id,
			TransEvClassMapping map, TObjectIntMap<Transition> mapTrans2Cost,
			TObjectIntMap<XEventClass> mapEvClass2Cost, TObjectIntMap<Transition> mapSync2Cost, Marking initialMarking,
			Marking finalMarking) {
		this(net, classes, c2id, map, new GenericMap2Int<>(mapTrans2Cost, 1), new GenericMap2Int<>(mapEvClass2Cost, 1), //
				new GenericMap2Int<>(mapSync2Cost, 0), initialMarking, finalMarking);
	}

	public BasicSyncProductFactory(Petrinet net, XEventClasses classes, TObjectIntMap<XEventClass> c2id,
			TransEvClassMapping map, Map<Transition, Integer> mapTrans2Cost, Map<XEventClass, Integer> mapEvClass2Cost,
			Marking initialMarking, Marking finalMarking) {
		this(net, classes, c2id, map, new GenericMap2Int<>(mapTrans2Cost, 1), new GenericMap2Int<>(mapEvClass2Cost, 1), //
				new GenericMap2Int<Transition>(0), initialMarking, finalMarking);
	}

	public BasicSyncProductFactory(Petrinet net, XEventClasses classes, TObjectIntMap<XEventClass> c2id,
			TransEvClassMapping map, TObjectIntMap<Transition> mapTrans2Cost,
			TObjectIntMap<XEventClass> mapEvClass2Cost, Marking initialMarking, Marking finalMarking) {
		this(net, classes, c2id, map, new GenericMap2Int<>(mapTrans2Cost, 1), new GenericMap2Int<>(mapEvClass2Cost, 1), //
				new GenericMap2Int<Transition>(0), initialMarking, finalMarking);
	}

	private BasicSyncProductFactory(Petrinet net, XEventClasses classes, TObjectIntMap<XEventClass> c2id,
			TransEvClassMapping map, GenericMap2Int<Transition> mapTrans2Cost,
			GenericMap2Int<XEventClass> mapEvClass2Cost, GenericMap2Int<Transition> mapSync2Cost,
			Marking initialMarking, Marking finalMarking) {

		this.c2id = c2id;
		this.classes = classes;

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
		c2t = new TIntObjectHashMap<>(this.classCount);
		for (XEventClass clazz : classes.getClasses()) {
			c2lmCost[c2id.get(clazz)] = mapEvClass2Cost.get(clazz);
		}

		transitions = net.getTransitions().size();
		t2mmCost = new TIntArrayList(transitions * 2);
		t2smCost = new TIntArrayList(transitions * 2);
		t2eid = new ObjectList<>(transitions * 2);
		t2type = new TByteArrayList(transitions * 2);
		t2name = new ObjectList<>(transitions * 2);
		t2input = new ObjectList<>(transitions * 2);
		t2output = new ObjectList<>(transitions * 2);
		t2transition = new Transition[transitions];

		places = net.getPlaces().size();
		p2name = new ObjectList<>(places * 2);
		TObjectIntMap<Place> p2id = new TObjectIntHashMap<>(net.getPlaces().size(), 0.75f, -1);
		//		t2id = new TObjectIntHashMap<>(net.getTransitions().size(), 0.75f, -1);

		// build list of move_model transitions
		Integer cost;
		Iterator<Transition> it = net.getTransitions().iterator();
		while (it.hasNext()) {
			Transition t = it.next();
			//			t2id.put(t, t2name.size());
			t2transition[t2name.size()] = t;

			// update mapping from event class to transitions
			XEventClass clazz = map.get(t);
			if (clazz != null) {
				TIntSet set = c2t.get(c2id.get(clazz));
				if (set == null) {
					set = new TIntHashSet(3);
					c2t.put(c2id.get(clazz), set);
				}
				set.add(t2name.size());
			}

			cost = mapTrans2Cost.get(t);
			t2mmCost.add(cost);
			cost = mapSync2Cost.get(t);
			t2smCost.add(cost);
			t2name.add(t.getLabel());
			t2eid.add(SyncProduct.NOEVENT);
			t2type.add(t.isInvisible() ? SyncProduct.TAU_MOVE : SyncProduct.MODEL_MOVE);

			TIntList input = new TIntArrayList(2 * net.getInEdges(t).size());
			for (PetrinetEdge<?, ?> e : net.getInEdges(t)) {
				Place p = (Place) e.getSource();
				int id = p2id.get(p);
				if (id == -1) {
					id = p2id.size();
					p2id.put(p, id);
					p2name.add(p.getLabel());
				}
				for (int w = 0; w < ((Arc) e).getWeight(); w++) {
					input.add(id);
				}
			}
			t2input.add(input.toArray());

			TIntList output = new TIntArrayList(2 * net.getOutEdges(t).size());
			for (PetrinetEdge<?, ?> e : net.getOutEdges(t)) {
				Place p = (Place) e.getTarget();
				int id = p2id.get(p);
				if (id == -1) {
					id = p2id.size();
					p2id.put(p, id);
					p2name.add(p.getLabel());
				}
				for (int w = 0; w < ((Arc) e).getWeight(); w++) {
					output.add(id);
				}
			}
			t2output.add(output.toArray());
		}

		// mark the initial places
		initMarking = new byte[p2name.size()];
		for (Place p : initialMarking) {
			int id = p2id.get(p);
			if (id >= 0) {
				initMarking[id]++;
			}
		}

		// indicate the desired final marking
		finMarking = new byte[p2name.size()];
		for (Place p : finalMarking) {
			int id = p2id.get(p);
			if (id >= 0) {
				finMarking[id]++;
			}
		}

	}

	public synchronized SyncProduct getSyncProductForEmptyTrace(ArrayList<? super Transition> transitionList) {
		return getLinearSyncProduct(new LinearTrace("Empty", 0), transitionList);
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

	public synchronized SyncProduct getSyncProduct(XTrace xTrace, ArrayList<? super Transition> transitionList,
			boolean partiallyOrderSameTimestamp) {
		String traceLabel = XConceptExtension.instance().extractName(xTrace);
		if (traceLabel == null) {
			traceLabel = "XTrace@" + Integer.toHexString(xTrace.hashCode());
		}
		if (partiallyOrderSameTimestamp) {
			PartiallyOrderedTrace trace = getPartiallyOrderedTrace(xTrace, traceLabel);
			// Do the ranking on this trace.
			return getPartiallyOrderedSyncProduct(trace, transitionList);
		} else {
			SyncProduct result = getLinearSyncProduct(getLinearTrace(xTrace, traceLabel), transitionList);
			//			Utils.toTpnSplitStartComplete(result, System.out);
			return result;
		}

	}

	private SyncProduct getLinearSyncProduct(LinearTrace trace, List<? super Transition> transitionList) {
		transitionList.clear();
		// for this trace, compute the log-moves
		// compute the sync moves
		TIntList ranks = new TIntArrayList();
		for (int t = 0; t < transitions; t++) {
			ranks.add(SyncProduct.NORANK);
		}

		for (int e = 0; e < trace.getSize(); e++) {
			//			XEventClass clazz = classes.getClassOf(trace.get(e));
			int cid = trace.get(e); // c2id.get(clazz);
			// add a place
			p2name.add("pe_" + e);
			// add log move
			t2name.add("e" + e + "(" + cid + ")");//clazz.toString());
			t2mmCost.add(c2lmCost[cid]);
			t2eid.add(new int[] { e });
			t2type.add(SyncProduct.LOG_MOVE);
			ranks.add(e);
			TIntSet set = c2t.get(cid);
			if (set != null) {
				TIntIterator it = set.iterator();
				while (it.hasNext()) {
					// add sync move
					int t = it.next();
					t2name.add(t2name.get(t) + " + e" + e + "(" + cid + ")");
					t2mmCost.add(t2smCost.get(t));
					t2eid.add(new int[] { e });
					t2type.add(SyncProduct.SYNC_MOVE);
					ranks.add(e);
				}
			}
		}
		if (trace.getSize() > 0) {
			p2name.add("pe_" + trace.getSize());
		}
		int[] pathLengths = new int[t2name.size()];
		Arrays.fill(pathLengths, 1);

		SyncProductImpl product = new SyncProductImpl(trace.getLabel(), //label
				this.classCount, // number of classes
				t2name.toArray(new String[t2name.size()]), //transition labels
				p2name.toArray(new String[p2name.size()]), // place labels
				t2eid.toArray(new int[t2eid.size()][]), //event numbers
				ranks.toArray(), // ranks
				pathLengths, t2type.toArray(), //types
				t2mmCost.toArray());

		int t = 0;
		for (; t < transitions; t++) {
			// first the model moves
			product.setInput(t, t2input.get(t));
			product.setOutput(t, t2output.get(t));
			transitionList.add(t2transition[t]);
		}

		for (int e = 0; e < trace.getSize(); e++) {
			// then the log moves
			//			XEventClass clazz = classes.getClassOf(trace.get(e));
			int cid = trace.get(e);// c2id.get(clazz);
			product.setInput(t, places + e);
			product.setOutput(t, places + e + 1);
			transitionList.add(null);
			t++;

			TIntSet set = c2t.get(cid);
			if (set != null) {
				TIntIterator it = set.iterator();
				while (it.hasNext()) {
					// add sync move
					int t2 = it.next();
					product.setInput(t, t2input.get(t2));
					product.setOutput(t, t2output.get(t2));

					product.addToInput(t, (places + e));
					product.addToOutput(t, (places + e + 1));

					transitionList.add(t2transition[t2]);
					t++;
				}
			}
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

	private SyncProduct getPartiallyOrderedSyncProduct(PartiallyOrderedTrace trace,
			List<? super Transition> transitionList) {
		transitionList.clear();

		int[] e2t = new int[trace.getSize()];

		// for this trace, compute the log-moves
		// compute the sync moves
		int[] outputPlaces = new int[trace.getSize()];

		for (int e = 0; e < trace.getSize(); e++) {
			//			XEventClass clazz = classes.getClassOf(trace.get(e));
			int cid = trace.get(e); // c2id.get(clazz);
			int[] predecessors = trace.getPredecessors(e);
			if (predecessors == null) {
				// initial place
				// add a place
				p2name.add("p_init-" + e);
			} else {
				for (int pi = 0; pi < predecessors.length; pi++) {
					// add a place
					// and record this as one of the output places of predecessors[pi]
					outputPlaces[predecessors[pi]] = p2name.size();
					p2name.add("p_" + predecessors[pi] + "-" + e);
				}
			}

			e2t[e] = t2name.size();
			// add log move
			t2name.add("e" + e + "(" + cid + ")");//clazz.toString());
			t2mmCost.add(c2lmCost[cid]);
			t2eid.add(new int[] { e });
			t2type.add(SyncProduct.LOG_MOVE);

			TIntSet set = c2t.get(cid);
			if (set != null) {
				TIntIterator it = set.iterator();
				while (it.hasNext()) {
					// add sync move
					int t = it.next();
					t2name.add(t2name.get(t) + ",e" + e + "(" + cid + ")");
					t2mmCost.add(t2smCost.get(t));
					t2eid.add(new int[] { e });
					t2type.add(SyncProduct.SYNC_MOVE);
				}
			}

		}

		for (int e = 0; e < outputPlaces.length; e++) {
			if (outputPlaces[e] == 0) {
				// no output place recorded.
				outputPlaces[e] = -p2name.size() - 1;
				p2name.add("p_" + e + "-output");
			}
		}

		int[] ranks = new int[t2eid.size()];
		Arrays.fill(ranks, SyncProduct.NORANK);
		int[] pathLengths = new int[t2name.size()];
		Arrays.fill(pathLengths, 1);

		SyncProductImpl product = new SyncProductImpl(trace.getLabel(), //label
				this.classCount, //number of event classes
				t2name.toArray(new String[t2name.size()]), //transition labels
				p2name.toArray(new String[p2name.size()]), // place labels
				t2eid.toArray(new int[t2eid.size()][]), //event numbers
				ranks, // ranks
				pathLengths, // pathlengths
				t2type.toArray(), //types
				t2mmCost.toArray());

		int t = 0;
		for (; t < transitions; t++) {
			// first the model moves
			product.setInput(t, t2input.get(t));
			product.setOutput(t, t2output.get(t));
			transitionList.add(t2transition[t]);
		}

		product.setInitialMarking(Arrays.copyOf(initMarking, p2name.size()));
		product.setFinalMarking(Arrays.copyOf(finMarking, p2name.size()));

		for (int e = 0; e < outputPlaces.length; e++) {
			if (outputPlaces[e] < 0) {
				product.addToFinalMarking(-outputPlaces[e] - 1);
			}
		}
		// TODO: Handle the sync product ranking properly. Currently, a random sequence
		// of events is ranked and the assumption is that events are ordered, i.e. that 
		// the predecessors of the event at index e are a index < e .

		int minRank = -1;
		int p = places;
		for (int e = 0; e < trace.getSize(); e++) {

			int cid = trace.get(e); // c2id.get(clazz);

			int[] predecessors = trace.getPredecessors(e);
			if (predecessors == null) {
				// initial place
				// add a place
				product.setRankOf(e2t[e], e);
				product.setInput(e2t[e], p);
				product.addToInitialMarking(p);
				p++;
			} else {
				for (int pi = 0; pi < predecessors.length; pi++) {
					// add a place
					if (product.getRankOf(e2t[predecessors[pi]]) == minRank) {
						product.setRankOf(e2t[e], ++minRank);
					}
					product.addToInput(e2t[e], p);
					product.addToOutput(e2t[predecessors[pi]], p);
					p++;
				}

			}
			transitionList.add(null);

			TIntSet set = c2t.get(cid);
			if (set != null) {
				TIntIterator it = set.iterator();
				while (it.hasNext()) {
					// add sync move
					int t2 = it.next();

					transitionList.add(t2transition[t2]);
				}
			}
		}
		for (int e = 0; e < trace.getSize(); e++) {
			if (outputPlaces[e] < 0) {
				product.addToOutput(e2t[e], -outputPlaces[e] - 1);
			}
			int cid = trace.get(e); // c2id.get(clazz);
			t++;
			TIntSet set = c2t.get(cid);
			if (set != null) {
				TIntIterator it = set.iterator();
				while (it.hasNext()) {
					// add sync move
					int t2 = it.next();
					product.setInput(t, t2input.get(t2));
					product.setOutput(t, t2output.get(t2));

					product.addToInput(t, product.getInput(e2t[e]));
					product.addToOutput(t, product.getOutput(e2t[e]));

					t++;
				}
			}
		}
		for (int e = 1; e < trace.getSize(); e++) {
			// for the log moves, put a test arc on the output of the predecessor which is marked
			// by either a log or a sync move.
			int[] predecessors = trace.getPredecessors(e);
			if (predecessors != null && predecessors[predecessors.length - 1] < e - 1) {
				if (outputPlaces[e - 1] < 0) {
					product.addToInput(e2t[e], -outputPlaces[e - 1] - 1);
					product.addToOutput(e2t[e], -outputPlaces[e - 1] - 1);
				} else {
					product.addToInput(e2t[e], outputPlaces[e - 1]);
					product.addToOutput(e2t[e], outputPlaces[e - 1]);
				}
			}
		}

		// trim to size;
		p2name.truncate(places);
		t2name.truncate(transitions);
		t2mmCost.remove(transitions, t2mmCost.size() - transitions);
		//		t2eid.remove(transitions, t2eid.size() - transitions);
		t2eid.truncate(transitions);
		t2type.remove(transitions, t2type.size() - transitions);

		return product;
	}

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
			ArrayList<? super Transition> transitionList) {
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
					for (int e : events) {
						stepTypes.add(StepTypes.LMGOOD);
						smm += replayer.getCostMM((Transition) transitionList.get(t));
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

	public SyncProduct getSyncProduct(XTrace xTrace, ArrayList<? super Transition> transitionList,
			boolean partiallyOrderSameTimestamp, boolean isPrefix) {
		String traceLabel = XConceptExtension.instance().extractName(xTrace);
		if (traceLabel == null) {
			traceLabel = "XTrace@" + Integer.toHexString(xTrace.hashCode());
		}
		if (partiallyOrderSameTimestamp) {
			PartiallyOrderedTrace trace = getPartiallyOrderedTrace(xTrace, traceLabel);
			// Do the ranking on this trace.
			return getPartiallyOrderedSyncProduct(trace, transitionList);
		} else {
			SyncProduct result = null;
			
			if (!isPrefix) {
				result = getLinearSyncProduct(getLinearTrace(xTrace, traceLabel), transitionList);	
			} else {
				result = getLinearSyncProductPrefix(getLinearTrace(xTrace, traceLabel), transitionList);
			}
			
			//			Utils.toTpnSplitStartComplete(result, System.out);
			return result;
		}
	}
	private SyncProduct getLinearSyncProductPrefix(LinearTrace trace, List<? super Transition> transitionList) {
		transitionList.clear();
		// for this trace, compute the log-moves
		// compute the sync moves
		TIntList ranks = new TIntArrayList();
		for (int t = 0; t < transitions; t++) {
			ranks.add(SyncProduct.NORANK);
		}

		for (int e = 0; e < trace.getSize(); e++) {
			//			XEventClass clazz = classes.getClassOf(trace.get(e));
			int cid = trace.get(e); // c2id.get(clazz);
			// add a place
			p2name.add("pe_" + e);
			// add log move
			t2name.add("e" + e + "(" + cid + ")");//clazz.toString());
			t2mmCost.add(c2lmCost[cid]);
			t2eid.add(new int[] { e });
			t2type.add(SyncProduct.LOG_MOVE);
			ranks.add(e);
			TIntSet set = c2t.get(cid);
			if (set != null) {
				TIntIterator it = set.iterator();
				while (it.hasNext()) {
					// add sync move
					int t = it.next();
					t2name.add(t2name.get(t) + " + e" + e + "(" + cid + ")");
					t2mmCost.add(t2smCost.get(t));
					t2eid.add(new int[] { e });
					t2type.add(SyncProduct.SYNC_MOVE);
					ranks.add(e);
				}
			}
		}
		if (trace.getSize() > 0) {
			p2name.add("pe_" + trace.getSize());
		}
		int[] pathLengths = new int[t2name.size()];
		Arrays.fill(pathLengths, 1);

		SyncProductPrefixImpl product = new SyncProductPrefixImpl(trace.getLabel(), //label
				this.classCount, // number of classes
				t2name.toArray(new String[t2name.size()]), //transition labels
				p2name.toArray(new String[p2name.size()]), // place labels
				t2eid.toArray(new int[t2eid.size()][]), //event numbers
				ranks.toArray(), // ranks
				pathLengths, t2type.toArray(), //types
				t2mmCost.toArray());

		int t = 0;
		for (; t < transitions; t++) {
			// first the model moves
			product.setInput(t, t2input.get(t));
			product.setOutput(t, t2output.get(t));
			transitionList.add(t2transition[t]);
		}

		for (int e = 0; e < trace.getSize(); e++) {
			// then the log moves
			//			XEventClass clazz = classes.getClassOf(trace.get(e));
			int cid = trace.get(e);// c2id.get(clazz);
			product.setInput(t, places + e);
			product.setOutput(t, places + e + 1);
			transitionList.add(null);
			t++;

			TIntSet set = c2t.get(cid);
			if (set != null) {
				TIntIterator it = set.iterator();
				while (it.hasNext()) {
					// add sync move
					int t2 = it.next();
					product.setInput(t, t2input.get(t2));
					product.setOutput(t, t2output.get(t2));

					product.addToInput(t, (places + e));
					product.addToOutput(t, (places + e + 1));

					transitionList.add(t2transition[t2]);
					t++;
				}
			}
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

		return product;
	}
}
