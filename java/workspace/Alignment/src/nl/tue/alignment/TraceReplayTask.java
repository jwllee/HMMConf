package nl.tue.alignment;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.Callable;

import org.deckfour.xes.extension.std.XConceptExtension;
import org.deckfour.xes.factory.XFactoryRegistry;
import org.deckfour.xes.model.XEvent;
import org.deckfour.xes.model.XTrace;
import org.processmining.plugins.replayer.replayresult.SyncReplayResult;

import gnu.trove.map.TObjectIntMap;
import nl.tue.alignment.Utils.Statistic;
import nl.tue.alignment.algorithms.ReplayAlgorithm;
import nl.tue.alignment.algorithms.ReplayAlgorithm.Debug;
import nl.tue.alignment.algorithms.implementations.AStar;
import nl.tue.alignment.algorithms.implementations.AStarLargeLP;
import nl.tue.alignment.algorithms.implementations.Dijkstra;
import nl.tue.alignment.algorithms.syncproduct.SyncProduct;
import nl.tue.astar.Trace;
import nl.tue.astar.util.ilp.LPMatrixException;

public class TraceReplayTask implements Callable<TraceReplayTask> {

	public static enum TraceReplayResult {
		FAILED, DUPLICATE, SUCCESS
	}

	// transient fields. Cleared after call();
	private transient Replayer replayer;
	private transient XTrace trace;
	private transient ReplayerParameters parameters;
	private transient SyncProduct product;
	private transient ReplayAlgorithm algorithm;

	// Available after "call()"
	private int original;
	private final int traceIndex;
	private SyncReplayResult srr;
	private TraceReplayResult result;
	private int traceLogMoveCost;

	// internal variables
	private int[] alignment;
	private int maximumNumberOfStates;
	private int[] eventsWithErrors;
	private final long preProcessTimeNanoseconds;
	private final int timeoutMilliseconds;

	public TraceReplayTask(Replayer replayer, ReplayerParameters parameters, int timeoutMilliseconds,
			int maximumNumberOfStates, long preProcessTimeNanoseconds, int... eventsWithErrors) {
		this.replayer = replayer;
		this.parameters = parameters;
		this.maximumNumberOfStates = maximumNumberOfStates;
		this.preProcessTimeNanoseconds = preProcessTimeNanoseconds;
		this.trace = XFactoryRegistry.instance().currentDefault().createTrace();
		XConceptExtension.instance().assignName(trace, "Empty");
		this.traceIndex = -1;
		this.timeoutMilliseconds = timeoutMilliseconds;
		this.eventsWithErrors = eventsWithErrors;
		Arrays.sort(eventsWithErrors);

	}

	@Deprecated
	public TraceReplayTask(Replayer replayer, ReplayerParameters parameters, int timeoutMilliseconds,
			int maximumNumberOfStates, int... eventsWithErrors) {
		this(replayer, parameters, timeoutMilliseconds, maximumNumberOfStates, 0, eventsWithErrors);
	}

	public TraceReplayTask(Replayer replayer, ReplayerParameters parameters, XTrace trace, int traceIndex,
			int timeoutMilliseconds, int maximumNumberOfStates, long preProcessTimeNanoseconds,
			int... eventsWithErrors) {
		this.replayer = replayer;
		this.parameters = parameters;
		this.trace = trace;
		this.traceIndex = traceIndex;
		this.timeoutMilliseconds = timeoutMilliseconds;
		this.maximumNumberOfStates = maximumNumberOfStates;
		this.preProcessTimeNanoseconds = preProcessTimeNanoseconds;
		this.eventsWithErrors = eventsWithErrors;
		Arrays.sort(eventsWithErrors);

	}

	@Deprecated
	public TraceReplayTask(Replayer replayer, ReplayerParameters parameters, XTrace trace, int traceIndex,
			int timeoutMilliseconds, int maximumNumberOfStates, int... eventsWithErrors) {
		this(replayer, parameters, trace, traceIndex, timeoutMilliseconds, maximumNumberOfStates, 0, eventsWithErrors);
	}

	private int getTraceCost(XTrace trace) {
		int cost = 0;
		for (XEvent e : trace) {
			cost += replayer.getCostLM(replayer.getEventClass(e));
		}
		return cost;
	}

	public TraceReplayTask call() throws LPMatrixException {
		if (replayer == null) {
			return this;
		}

		Trace traceAsList = this.replayer.factory.getTrace(trace, parameters.partiallyOrderEvents);
		this.traceLogMoveCost = getTraceCost(trace);

		if (this.replayer.mergeDuplicateTraces) {
			synchronized (this.replayer.trace2FirstIdenticalTrace) {
				original = this.replayer.trace2FirstIdenticalTrace.get(traceAsList);
				if (original < 0) {
					this.replayer.trace2FirstIdenticalTrace.put(traceAsList, traceIndex);
				}
			}
		} else {
			original = -1;
		}
		if (original < 0) {
			//			System.out.println("Starting trace: " + traceIndex);
			ArrayList<Object> transitionList = new ArrayList<>();
			long pt;
			synchronized (this.replayer.factory) {
				long startSP = System.nanoTime();
				// product = this.replayer.factory.getSyncProduct(trace, transitionList, parameters.partiallyOrderEvents);
				product = this.replayer.factory.getSyncProduct(trace, transitionList, 
						parameters.partiallyOrderEvents, parameters.isPrefix);
				long endSP = System.nanoTime();
				// compute the pre-process time based on the fixed overhead, plus time for this replayer...
				pt = preProcessTimeNanoseconds + (endSP - startSP);
			}

			if (product != null) {
				if (parameters.debug == Debug.DOT) {
					Utils.toDot(product, ReplayAlgorithm.Debug.getOutputStream());
				}

				algorithm = getAlgorithm(product);

				algorithm.putStatistic(Statistic.PREPROCESSTIME, (int) (pt / 1000));
				algorithm.putStatistic(Statistic.CONSTRAINTSETSIZE, replayer.getConstraintSetSize());

				alignment = algorithm.run(this.replayer.getProgress(), timeoutMilliseconds, maximumNumberOfStates,
						parameters.costUpperBound);

				if (parameters.debug == Debug.DOT) {
					Utils.toDot(product, alignment, ReplayAlgorithm.Debug.getOutputStream());
				}
				TObjectIntMap<Statistic> stats = algorithm.getStatistics();

				srr = this.replayer.factory.toSyncReplayResult(replayer, product, stats, alignment, trace, traceIndex,
						transitionList);
				this.replayer.getProgress().inc();
				result = TraceReplayResult.SUCCESS;
			} else {
				this.replayer.getProgress().inc();
				result = TraceReplayResult.FAILED;
			}
		} else {
			this.replayer.getProgress().inc();
			result = TraceReplayResult.DUPLICATE;
		}

		replayer = null;
		trace = null;
		parameters = null;
		algorithm = null;
		product = null;

		return this;
	}

	private ReplayAlgorithm getAlgorithm(SyncProduct product) throws LPMatrixException {
		switch (parameters.algorithm) {
			case ASTAR :
				if (parameters.buildFullStatespace) {
					return new AStar.Full(product, parameters.moveSort, parameters.queueSort, parameters.preferExact, //
							parameters.useInt, parameters.debug);
				} else {
					return new AStar(product, parameters.moveSort, parameters.queueSort, parameters.preferExact, //
							parameters.useInt, parameters.debug);
				}
			case INCREMENTALASTAR :
				if (parameters.buildFullStatespace) {
					if (parameters.initialSplits > 0 && eventsWithErrors.length == 0) {
						return new AStarLargeLP.Full(product, parameters.moveSort, parameters.useInt,
								parameters.initialSplits, parameters.debug);
					} else {
						return new AStarLargeLP.Full(product, parameters.moveSort, parameters.useInt, parameters.debug,
								eventsWithErrors);
					}
				} else {
					if (parameters.initialSplits > 0 && eventsWithErrors.length == 0) {
						return new AStarLargeLP(product, parameters.moveSort, parameters.useInt,
								parameters.initialSplits, parameters.debug);
					} else {
						return new AStarLargeLP(product, parameters.moveSort, parameters.useInt, parameters.debug,
								eventsWithErrors);
					}
				}
			case DIJKSTRA :
				if (parameters.buildFullStatespace) {
					return new Dijkstra.Full(product, parameters.moveSort, parameters.queueSort, parameters.debug);
				} else {
					return new Dijkstra(product, parameters.moveSort, parameters.queueSort, parameters.debug);
				}
		}
		assert false;
		return null;
	}

	public TraceReplayResult getResult() {
		return result;
	}

	public SyncReplayResult getSuccesfulResult() {
		return srr;
	}

	public int getOriginalTraceIndex() {
		return original;
	}

	public int getTraceIndex() {
		return traceIndex;
	}

	public int getTraceLogMoveCost() {
		return traceLogMoveCost;
	}
}