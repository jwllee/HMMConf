package nl.tue.alignment;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.deckfour.xes.classification.XEventClass;
import org.deckfour.xes.classification.XEventClasses;
import org.deckfour.xes.model.XEvent;
import org.deckfour.xes.model.XLog;
import org.deckfour.xes.model.XTrace;
import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.graphbased.directed.petrinet.elements.Transition;
import org.processmining.models.semantics.petrinet.Marking;
import org.processmining.plugins.connectionfactories.logpetrinet.TransEvClassMapping;
import org.processmining.plugins.petrinet.replayresult.PNRepResult;
import org.processmining.plugins.petrinet.replayresult.PNRepResultImpl;
import org.processmining.plugins.replayer.replayresult.SyncReplayResult;

import gnu.trove.list.TIntList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.TObjectIntMap;
import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;
import nl.tue.alignment.ReplayerParameters.Algorithm;
import nl.tue.alignment.TraceReplayTask.TraceReplayResult;
import nl.tue.alignment.Utils.Statistic;
import nl.tue.alignment.algorithms.ReplayAlgorithm.Debug;
import nl.tue.alignment.algorithms.constraints.ConstraintSet;
import nl.tue.alignment.algorithms.syncproduct.BasicSyncProductFactory;
import nl.tue.alignment.algorithms.syncproduct.ReducedSyncProductFactory;
import nl.tue.alignment.algorithms.syncproduct.SyncProductFactory;
import nl.tue.astar.Trace;

public class Replayer {

	public static final String MAXMODELMOVECOST = "Model move cost empty trace";
	public static final String TRACEEXITCODE = "Exit code of alignment for trace";
	public static final String MEMORYUSED = "Approximate memory used (kb)";
	public static final String PREPROCESSTIME = "Pre-processing time (ms)";
	public static final String HEURISTICSCOMPUTED = "Number of LPs solved";

	final TObjectIntMap<Trace> trace2FirstIdenticalTrace;

	private final ReplayerParameters parameters;
	//	private final XLog log;
	private final Map<XEventClass, Integer> costLM;
	final SyncProductFactory<?> factory;
	final XEventClasses classes;
	final private Map<Transition, Integer> costMM;
	private Progress progress;
	final boolean mergeDuplicateTraces;

	private TObjectIntMap<XEventClass> class2id;

	private ConstraintSet constraintSet;

	public Replayer(Petrinet net, Marking initialMarking, Marking finalMarking, XEventClasses classes,
			Map<Transition, Integer> costMOS, Map<XEventClass, Integer> costMOT, TransEvClassMapping mapping,
			boolean mergeDuplicateTraces) {
		this(new ReplayerParameters.Default(), net, initialMarking, finalMarking, classes, costMOS, costMOT, null,
				mapping, mergeDuplicateTraces);
	}

	public Replayer(Petrinet net, Marking initialMarking, Marking finalMarking, XEventClasses classes,
			TransEvClassMapping mapping, boolean mergeDuplicateTraces) {
		this(new ReplayerParameters.Default(), net, initialMarking, finalMarking, classes, null, null, null, mapping,
				mergeDuplicateTraces);
	}

	public Replayer(ReplayerParameters parameters, Petrinet net, Marking initialMarking, Marking finalMarking,
			XEventClasses classes, Map<Transition, Integer> costMOS, Map<XEventClass, Integer> costMOT,
			TransEvClassMapping mapping, boolean mergeDuplicateTraces) {
		this(parameters, net, initialMarking, finalMarking, classes, costMOS, costMOT, null, mapping,
				mergeDuplicateTraces);
	}

	public Replayer(ReplayerParameters parameters, Petrinet net, Marking initialMarking, Marking finalMarking,
			XEventClasses classes, TransEvClassMapping mapping, boolean mergeDuplicateTraces) {
		this(parameters, net, initialMarking, finalMarking, classes, null, null, null, mapping, mergeDuplicateTraces);
	}

	public Replayer(ReplayerParameters parameters, Petrinet net, Marking initialMarking, Marking finalMarking,
			XEventClasses classes, Map<Transition, Integer> costMM, Map<XEventClass, Integer> costLM,
			Map<Transition, Integer> costSM, TransEvClassMapping mapping, boolean mergeDuplicateTraces) {
		this.parameters = parameters;
		this.classes = classes;
		this.mergeDuplicateTraces = mergeDuplicateTraces;
		if (costMM == null) {
			costMM = new HashMap<>();
			for (Transition t : net.getTransitions()) {
				if (t.isInvisible()) {
					costMM.put(t, 0);
				} else {
					costMM.put(t, 1);
				}
			}
		}
		if (costSM == null) {
			costSM = new HashMap<>();
			for (Transition t : net.getTransitions()) {
				costSM.put(t, 0);
			}
		}
		if (costLM == null) {
			costLM = new HashMap<>();
			for (XEventClass clazz : classes.getClasses()) {
				costLM.put(clazz, 1);
			}
		}
		this.costMM = costMM;
		this.costLM = costLM;
		class2id = Utils.createClass2ID(classes);
		if (parameters.preProcessUsingPlaceBasedConstraints) {
			constraintSet = new ConstraintSet(net, initialMarking, classes, class2id, mapping);
		} else {
			constraintSet = null;
		}

		if (parameters.maxReducedSequenceLength > 1) {
			factory = new ReducedSyncProductFactory(net, classes, class2id, mapping, costMM, costLM, costSM,
					initialMarking, finalMarking, parameters.maxReducedSequenceLength);
		} else {
			factory = new BasicSyncProductFactory(net, classes, class2id, mapping, costMM, costLM, costSM,
					initialMarking, finalMarking);
		}

		if (mergeDuplicateTraces) {
			trace2FirstIdenticalTrace = new TObjectIntHashMap<>(10, 0.7f, -1);
		} else {
			trace2FirstIdenticalTrace = null;
		}
	}

	public PNRepResult computePNRepResult(Progress progress, XLog log) throws InterruptedException, ExecutionException {
		this.progress = progress;

		if (parameters.debug == Debug.STATS) {
			parameters.debug.print(Debug.STATS, "SP label");
			for (Statistic s : Statistic.values()) {
				parameters.debug.print(Debug.STATS, Utils.SEP);
				parameters.debug.print(Debug.STATS, s.toString());
			}
			parameters.debug.print(Debug.STATS, Utils.SEP + "max Memory (MB)");
			parameters.debug.print(Debug.STATS, Utils.SEP + "total Memory (MB)");
			parameters.debug.print(Debug.STATS, Utils.SEP + "free Memory (MB)");

			if (parameters.algorithm == Algorithm.INCREMENTALASTAR) {
				parameters.debug.print(Debug.STATS, Utils.SEP + "Splitpoints");
			}

			parameters.debug.println(Debug.STATS);

		}

		// compute timeout per trace
		int timeoutMilliseconds = (int) ((10.0 * parameters.timeoutMilliseconds) / (log.size() + 1));
		timeoutMilliseconds = Math.min(timeoutMilliseconds, parameters.timeoutMilliseconds);

		ExecutorService service;
		//		if (parameters.algorithm == Algorithm.ASTAR) {
		//			// multi-threading is handled inside ASTAR
		//			service = Executors.newFixedThreadPool(1);
		//		} else {
		service = Executors.newFixedThreadPool(parameters.nThreads);
		//		}
		progress.setMaximum(log.size() + 1);

		List<Future<TraceReplayTask>> resultList = new ArrayList<>();

		TraceReplayTask tr = new TraceReplayTask(this, parameters, timeoutMilliseconds,
				parameters.maximumNumberOfStates, 0l);
		resultList.add(service.submit(tr));

		int t = 0;
		for (XTrace trace : log) {
			//			if (traceToInclude.length > 0
			//					&& !XConceptExtension.instance().extractName(trace).equals(traceToInclude[0])) {
			//				continue;
			//			}

			TIntList errorEvents = new TIntArrayList(trace.size());
			long preprocessTime = 0;
			if (constraintSet != null) {
				long start = System.nanoTime();
				// pre-process the trace
				constraintSet.reset();
				for (int e = 0; e < trace.size(); e++) {
					int label = class2id.get(classes.getClassOf(trace.get(e)));
					if (!constraintSet.satisfiedAfterOccurence(label)) {
						//						if (e > 0) {
						//							errorEvents.add((int) (e));
						//						}
						errorEvents.add((int) (e + 1));
					}
				}
				//				System.out.println("Splitpoints:" + errorEvents.toString());
				preprocessTime = (System.nanoTime() - start);
			}
			tr = new TraceReplayTask(this, parameters, trace, t, timeoutMilliseconds, parameters.maximumNumberOfStates,
					preprocessTime, errorEvents.toArray());
			resultList.add(service.submit(tr));
			t++;
		}

		service.shutdown();
		return mergeResults(resultList);
	}

	public PNRepResult mergeResults(List<Future<TraceReplayTask>> resultList)
			throws InterruptedException, ExecutionException {

		Iterator<Future<TraceReplayTask>> itResult = resultList.iterator();
		TraceReplayTask tr;

		// get the alignment of the empty trace
		int maxModelMoveCost;
		TraceReplayTask traceReplay = itResult.next().get();
		if (traceReplay.getResult() == TraceReplayResult.SUCCESS) {
			maxModelMoveCost = (int) Math
					.round(traceReplay.getSuccesfulResult().getInfo().get(PNRepResult.RAWFITNESSCOST));
			itResult.remove();
		} else if (traceReplay.getResult() == TraceReplayResult.DUPLICATE) {
			assert false;
			maxModelMoveCost = 0;
		} else {
			maxModelMoveCost = 0;
		}

		TIntObjectMap<SyncReplayResult> result = new TIntObjectHashMap<>(10, 0.5f, -1);
		// process further changes
		while (itResult.hasNext() && !isCancelled()) {
			tr = itResult.next().get();
			int traceCost = tr.getTraceLogMoveCost();

			if (tr.getResult() == TraceReplayResult.SUCCESS) {

				SyncReplayResult srr = tr.getSuccesfulResult();
				srr.addInfo(PNRepResult.TRACEFITNESS,
						1 - (srr.getInfo().get(PNRepResult.RAWFITNESSCOST) / (maxModelMoveCost + traceCost)));
				result.put(tr.getTraceIndex(), srr);
				//				System.out.println("Success: " + tr.getTraceIndex());

			} else if (tr.getResult() == TraceReplayResult.DUPLICATE) {
				// skip
			} else {
				// FAILURE TO COMPUTE ALIGNMENT
				System.err.println("Failure: " + tr.getTraceIndex());
			}
		}
		itResult = resultList.iterator();
		// skip empty trace
		itResult.next();
		while (itResult.hasNext() && !isCancelled()) {
			tr = itResult.next().get();
			if (tr.getResult() == TraceReplayResult.DUPLICATE) {
				SyncReplayResult srr = result.get(tr.getOriginalTraceIndex());
				srr.addNewCase(tr.getTraceIndex());
			}
		}

		PNRepResultImpl pnRepResult = new PNRepResultImpl(result.valueCollection());
		pnRepResult.addInfo(MAXMODELMOVECOST, Double.toString(maxModelMoveCost));
		return pnRepResult;

	}

	private boolean isCancelled() {
		return getProgress().isCanceled();
	}

	public int getCostLM(XEventClass classOf) {
		return costLM != null && costLM.containsKey(classOf) ? costLM.get(classOf) : 1;
	}

	public int getCostMM(Transition transition) {
		return costMM != null && costMM.containsKey(transition) ? costMM.get(transition) : 1;
	}

	public Progress getProgress() {
		return progress == null ? Progress.INVISIBLE : progress;
	}

	public int getConstraintSetSize() {
		return constraintSet == null ? 0 : constraintSet.size();
	}

	public XEventClass getEventClass(XEvent e) {
		return classes.getClassOf(e);
	}

}
