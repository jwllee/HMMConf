package nl.tue.alignment;

import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.deckfour.xes.classification.XEventClass;
import org.deckfour.xes.classification.XEventClasses;
import org.deckfour.xes.classification.XEventClassifier;
import org.deckfour.xes.info.XLogInfo;
import org.deckfour.xes.info.XLogInfoFactory;
import org.deckfour.xes.model.XLog;
import org.deckfour.xes.model.XTrace;
import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.graphbased.directed.petrinet.PetrinetGraph;
import org.processmining.models.graphbased.directed.petrinet.elements.Transition;
import org.processmining.models.semantics.petrinet.Marking;
import org.processmining.plugins.connectionfactories.logpetrinet.TransEvClassMapping;
import org.processmining.plugins.petrinet.replayresult.PNRepResult;
import org.processmining.plugins.replayer.replayresult.SyncReplayResult;

import nl.tue.alignment.algorithms.ReplayAlgorithm.Debug;

/**
 * To use this class for experiments, the following code snippet can be used. It
 * is required for the provided log that the first trace is an empty trace:
 *
 * <code>
	public static void doLogReplay() {
		TraceByTraceAlignment traceByTraceAlignment = new TraceByTraceAlignment(net, initialMarking, finalMarking, log, classes, mapping);
		
		List<Future<TraceReplayTask>> list = new ArrayList<>(log.size());
		for (int i=0; i< log.size(); i++) {
			list.add(traceByTraceAlignment.doReplay(i, 60*10*000), eventsWithErrors);
		}
		
		PNRepResult repResult = traceByTraceAlignment.merge(list);
	}
 * </code>
 * 
 * @author bfvdonge
 *
 */
public class TraceByTraceAlignment {

	private Petrinet net;
	private Marking initialMarking;
	private Marking finalMarking;
	private XEventClasses classes;
	private TransEvClassMapping mapping;
	private ReplayerParameters.IncrementalAStar parameters;
	private Replayer replayer;

	/**
	 * Setup the trace-by-trace replayer using default parameters for the given net
	 * and log with a default, label-based mapping.
	 * 
	 * @param net
	 * @param initialMarking
	 * @param finalMarking
	 * @param classes
	 * @param mapping
	 */
	public TraceByTraceAlignment(Petrinet net, Marking initialMarking, Marking finalMarking, XEventClasses classes,
			TransEvClassMapping mapping) {
		this.net = net;
		this.initialMarking = initialMarking;
		this.finalMarking = finalMarking;
		this.classes = classes;
		this.mapping = mapping;

		setupParameters();
	}

	private void setupParameters() {

		// number of threads (irrelevant for trace by trace computations)
		int threads = 1;
		// timeout 30 sec per trace minutes
		int timeout = 30 * 1000 / 10;
		// no maximum state count
		int maxNumberOfStates = Integer.MAX_VALUE;
		// move sorting (should be false for incremental alignments)
		boolean moveSort = false;
		// use integers in linear programs (false for faster alignments)
		boolean useInt = false;
		// use partial orders (should be false for incremental alignments with pre-set splitpoints)
		boolean partialOrder = false;
		// upper bound for costs.
		int costUpperBound = Integer.MAX_VALUE;

		parameters = new ReplayerParameters.IncrementalAStar(moveSort, threads, useInt, Debug.NONE, timeout,
				maxNumberOfStates, costUpperBound, partialOrder, false);
		replayer = new Replayer(parameters, net, initialMarking, finalMarking, classes, mapping, false);

	}

	/**
	 * returns a future to allow for normal merging procedures, but computation is
	 * synchronously. Collect them in a list for later merging. When merging, the
	 * result is already available.
	 * 
	 * @param traceIndex
	 * @param timeoutMilliseconds
	 * @param eventsWithErrors
	 * @return
	 * @throws TimeoutException
	 * @throws ExecutionException
	 * @throws InterruptedException
	 */
	public Future<TraceReplayTask> doReplay(XTrace trace, int traceIndex, int timeoutMilliseconds,
			long preProcessTimeNanoseconds, int... eventsWithErrors)
			throws InterruptedException, ExecutionException, TimeoutException {

		TraceReplayTask task = new TraceReplayTask(replayer, parameters, trace, traceIndex, timeoutMilliseconds,
				parameters.maximumNumberOfStates, preProcessTimeNanoseconds, eventsWithErrors);

		FutureTask<TraceReplayTask> futureTask = new FutureTask<>(task);
		futureTask.run();
		futureTask.get(4 * timeoutMilliseconds / 3, TimeUnit.MILLISECONDS);
		return futureTask;
	}

	/**
	 * merge the future's
	 * 
	 * @param resultList
	 * @return
	 * @throws InterruptedException
	 * @throws ExecutionException
	 */
	public PNRepResult merge(List<Future<TraceReplayTask>> resultList) throws InterruptedException, ExecutionException {

		PNRepResult result = replayer.mergeResults(resultList);

		int cost = (int) Double.parseDouble((String) result.getInfo().get(Replayer.MAXMODELMOVECOST));
		int timeout = 0;
		int time = 0;
		int mem = 0;
		for (SyncReplayResult res : result) {
			cost += res.getInfo().get(PNRepResult.RAWFITNESSCOST);
			timeout += res.getInfo().get(Replayer.TRACEEXITCODE).intValue() != 1 ? 1 : 0;
			time += res.getInfo().get(PNRepResult.TIME).intValue();
			mem = Math.max(mem, res.getInfo().get(Replayer.MEMORYUSED).intValue());
		}
		System.out.print(time + ",");
		System.out.print(mem + ",");
		System.out.print(timeout + ",");
		System.out.print(cost + ",");

		System.out.println();
		System.out.flush();

		return result;
	}

	/**
	 * Constructs a default, label-based mapping
	 * 
	 * @param net
	 * @param log
	 * @param dummyEvClass
	 * @param eventClassifier
	 * @return
	 */
	private TransEvClassMapping constructMapping(PetrinetGraph net, XLog log, XEventClass dummyEvClass,
			XEventClassifier eventClassifier) {
		TransEvClassMapping mapping = new TransEvClassMapping(eventClassifier, dummyEvClass);

		XLogInfo summary = XLogInfoFactory.createLogInfo(log, eventClassifier);

		for (Transition t : net.getTransitions()) {
			for (XEventClass evClass : summary.getEventClasses().getClasses()) {
				String id = evClass.getId();

				// map transitions and event classes based on label
				if (t.getLabel().equals(id)) {
					mapping.put(t, evClass);
					break;
				}
			}

			//			if (!mapped && !t.isInvisible()) {
			//				mapping.put(t, dummyEvClass);
			//			}

		}

		return mapping;
	}

}
